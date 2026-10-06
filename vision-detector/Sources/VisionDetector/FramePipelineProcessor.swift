import Foundation
import CoreVideo
import CoreMedia
import CoreGraphics
import Vision
import QuartzCore

/// Configuration for the frame pipeline processor
struct ProcessorConfig {
    let width: Int
    let height: Int
    let pixelFormat: String  // "bgra", "rgba", "rgb24"
    let fps: Double
    let maxHands: Int
    let minConfidence: Float
    let cameraId: String
    let detectHands: Bool
    let detectBody: Bool
    let renderOverlay: Bool
    let useMetal: Bool  // Use Metal GPU rendering for overlay
    let asyncDetection: Bool  // Run detection async (non-blocking, uses stale results)
    let showStats: Bool  // Collect stats (for output and/or overlay)
    let renderStatsOverlay: Bool  // Render stats overlay onto frames
    let landmarksOutPath: String?
    let statsOutPath: String?  // Output stats JSONL to file

    var bytesPerPixel: Int {
        switch pixelFormat {
        case "rgb24": return 3
        default: return 4  // bgra, rgba
        }
    }

    var frameSize: Int {
        width * height * bytesPerPixel
    }
}

/// Processes video frames from stdin, runs detection, renders overlay, outputs to stdout
final class FramePipelineProcessor {
    private let config: ProcessorConfig
    private let handRequest: VNDetectHumanHandPoseRequest
    private let bodyRequest: VNDetectHumanBodyPoseRequest

    private var isRunning = false
    private var frameNumber: Int = 0
    private let startTime: CFTimeInterval

    private let landmarksHandle: FileHandle
    private let statsHandle: FileHandle?
    private let inputHandle = FileHandle.standardInput
    private let outputHandle = FileHandle.standardOutput

    // Current detection results (updated async, read sync)
    private var currentHands: [HandLandmarks] = []
    private var currentBody: BodyLandmarks? = nil
    private let detectionLock = NSLock()

    // Async detection queue and state
    private let detectionQueue = DispatchQueue(label: "vision.detection", qos: .userInitiated)
    private var pendingDetection = false
    private let pendingLock = NSLock()

    // Stats tracker (optional)
    private var stats: PipelineStats?

    // Metal renderer (optional, falls back to CGContext if unavailable)
    private var metalRenderer: MetalOverlayRenderer?

    init(config: ProcessorConfig) {
        self.config = config
        self.startTime = CACurrentMediaTime()

        // Setup hand detection
        self.handRequest = VNDetectHumanHandPoseRequest()
        self.handRequest.maximumHandCount = config.maxHands

        // Setup body detection
        self.bodyRequest = VNDetectHumanBodyPoseRequest()

        // Setup landmarks output
        if let path = config.landmarksOutPath {
            FileManager.default.createFile(atPath: path, contents: nil)
            self.landmarksHandle = FileHandle(forWritingAtPath: path) ?? FileHandle.standardError
        } else {
            self.landmarksHandle = FileHandle.standardError
        }

        // Setup stats output
        if let path = config.statsOutPath {
            FileManager.default.createFile(atPath: path, contents: nil)
            self.statsHandle = FileHandle(forWritingAtPath: path)
        } else {
            self.statsHandle = nil
        }

        // Initialize Metal renderer if requested
        if config.useMetal {
            self.metalRenderer = MetalOverlayRenderer()
            if metalRenderer == nil {
                fputs("Warning: Metal renderer unavailable, falling back to CGContext\n", stderr)
            }
        }

        // Initialize stats tracker if requested
        if config.showStats {
            self.stats = PipelineStats()
        }
    }

    func run() {
        isRunning = true
        var inputBuffer = Data(capacity: config.frameSize * 2)

        fputs("Reading frames from stdin (\(config.frameSize) bytes per frame)...\n", stderr)

        while isRunning {
            autoreleasepool {
                // Read data from stdin
                let chunk = inputHandle.readData(ofLength: config.frameSize - inputBuffer.count)
                if chunk.isEmpty {
                    // EOF
                    isRunning = false
                    return
                }

                inputBuffer.append(chunk)

                // Process complete frames
                while inputBuffer.count >= config.frameSize {
                    let frameData = Data(inputBuffer.prefix(config.frameSize))
                    inputBuffer = Data(inputBuffer.dropFirst(config.frameSize))

                    processFrame(frameData)
                }
            }
        }

        fputs("Processed \(frameNumber) frames.\n", stderr)

        // Print stats summary
        stats?.printSummary()
    }

    func stop() {
        isRunning = false
    }

    private func processFrame(_ frameData: Data) {
        // Start timing for stats
        let frameStartTime = stats?.startFrame()

        let currentFrameNum = frameNumber
        frameNumber += 1

        // Create pixel buffer from input data
        guard let inputBuffer = createPixelBuffer(from: frameData) else {
            fputs("Warning: Failed to create pixel buffer for frame \(currentFrameNum)\n", stderr)
            // Pass through unchanged
            outputHandle.write(frameData)
            return
        }

        if config.asyncDetection {
            // Async mode: start detection in background, render immediately with stale results
            runDetectionAsync(on: inputBuffer, frameNumber: currentFrameNum)
        } else {
            // Sync mode: wait for detection to complete before rendering
            runDetection(on: inputBuffer, frameNumber: currentFrameNum)
        }

        // Render overlay if enabled (uses latest available detection results)
        let outputData: Data
        if config.renderOverlay {
            outputData = renderOverlayAndExtract(inputBuffer)
        } else {
            outputData = frameData
        }

        // Update stats with current detection results
        if let startTime = frameStartTime {
            detectionLock.lock()
            let hands = currentHands
            let body = currentBody
            detectionLock.unlock()
            stats?.endFrame(startTime: startTime, hands: hands, body: body)

            // Output stats JSONL if configured
            if let statsHandle = statsHandle, let stats = stats {
                let statsFrame = stats.toStatsFrame(frameNumber: currentFrameNum, cameraId: config.cameraId)
                if let jsonLine = statsFrame.toJSONLine() {
                    let data = (jsonLine + "\n").data(using: .utf8)!
                    statsHandle.write(data)
                }
            }
        }

        // Write output frame
        outputHandle.write(outputData)
    }

    private func runDetectionAsync(on pixelBuffer: CVPixelBuffer, frameNumber: Int) {
        // Check if detection is already pending
        pendingLock.lock()
        if pendingDetection {
            pendingLock.unlock()
            return  // Skip this frame's detection, use stale results
        }
        pendingDetection = true
        pendingLock.unlock()

        // Copy pixel buffer data for async processing (pixel buffer may be reused)
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dataSize = bytesPerRow * height

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            pendingLock.lock()
            pendingDetection = false
            pendingLock.unlock()
            return
        }

        let dataCopy = Data(bytes: baseAddress, count: dataSize)
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

        // Run detection on background queue
        detectionQueue.async { [weak self] in
            guard let self = self else { return }

            // Create new pixel buffer from copied data
            if let asyncBuffer = self.createPixelBufferForAsync(from: dataCopy, width: width, height: height, bytesPerRow: bytesPerRow) {
                self.runDetection(on: asyncBuffer, frameNumber: frameNumber)
            }

            self.pendingLock.lock()
            self.pendingDetection = false
            self.pendingLock.unlock()
        }
    }

    private func createPixelBufferForAsync(from data: Data, width: Int, height: Int, bytesPerRow: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?

        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }

        let destBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        data.withUnsafeBytes { srcPtr in
            guard let src = srcPtr.baseAddress else { return }
            if destBytesPerRow == bytesPerRow {
                memcpy(baseAddress, src, data.count)
            } else {
                // Handle stride mismatch
                let dst = baseAddress.assumingMemoryBound(to: UInt8.self)
                let srcBytes = src.assumingMemoryBound(to: UInt8.self)
                let copyWidth = min(bytesPerRow, destBytesPerRow)
                for y in 0..<height {
                    memcpy(dst + y * destBytesPerRow, srcBytes + y * bytesPerRow, copyWidth)
                }
            }
        }

        return buffer
    }

    private func runDetection(on pixelBuffer: CVPixelBuffer, frameNumber: Int) {
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )

        var requests: [VNRequest] = []
        if config.detectHands {
            requests.append(handRequest)
        }
        if config.detectBody {
            requests.append(bodyRequest)
        }

        do {
            try handler.perform(requests)

            // Extract hand landmarks
            var hands: [HandLandmarks] = []
            if config.detectHands, let observations = handRequest.results {
                for observation in observations {
                    if let landmarks = try? HandLandmarks(from: observation) {
                        hands.append(landmarks)
                    }
                }
            }

            // Extract body landmarks
            var body: BodyLandmarks? = nil
            if config.detectBody, let observations = bodyRequest.results, let first = observations.first {
                body = try? BodyLandmarks(from: first)
            }

            // Update current detection (thread-safe)
            detectionLock.lock()
            currentHands = hands
            currentBody = body
            detectionLock.unlock()

            // Output landmarks
            let timestamp = CACurrentMediaTime() - startTime
            let frame = DetectionFrame(
                timestamp: timestamp,
                frameNumber: frameNumber,
                sourceCamera: config.cameraId,
                hands: hands,
                body: body
            )

            if let jsonLine = frame.toJSONLine() {
                let data = (jsonLine + "\n").data(using: .utf8)!
                landmarksHandle.write(data)
            }

        } catch {
            // Skip detection on error, pass frame through
            fputs("Warning: Detection failed for frame \(frameNumber): \(error.localizedDescription)\n", stderr)
        }
    }

    private func renderOverlayAndExtract(_ pixelBuffer: CVPixelBuffer) -> Data {
        // Get current detection results
        detectionLock.lock()
        let hands = currentHands
        let body = currentBody
        detectionLock.unlock()

        // Use Metal renderer if available
        if let metalRenderer = metalRenderer {
            metalRenderer.renderOverlay(
                onto: pixelBuffer,
                hands: hands,
                body: body,
                minConfidence: config.minConfidence
            )
            // Render stats overlay if enabled
            if config.renderStatsOverlay, let stats = stats {
                let height = CVPixelBufferGetHeight(pixelBuffer)
                stats.renderOverlay(onto: pixelBuffer, atX: 10, atY: height - stats.overlayHeight - 10)
            }
            return extractFrameData(from: pixelBuffer)
        }

        // Fallback to CGContext rendering
        return renderOverlayCGContext(pixelBuffer, hands: hands, body: body)
    }

    private func renderOverlayCGContext(_ pixelBuffer: CVPixelBuffer, hands: [HandLandmarks], body: BodyLandmarks?) -> Data {
        // Lock the pixel buffer for writing
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return extractFrameData(from: pixelBuffer)
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        // Create CGContext for drawing
        guard let context = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            return extractFrameData(from: pixelBuffer)
        }

        // Flip context to match pixel buffer coordinate system
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)

        // Draw hands
        for (fingerIndex, hand) in hands.enumerated() {
            drawHand(hand, in: context, width: width, height: height, colorIndex: fingerIndex)
        }

        // Draw body
        if let body = body {
            drawBody(body, in: context, width: width, height: height)
        }

        // Render stats overlay if enabled
        if config.renderStatsOverlay, let stats = stats {
            stats.renderOverlay(onto: pixelBuffer, atX: 10, atY: height - stats.overlayHeight - 10)
        }

        return extractFrameData(from: pixelBuffer)
    }

    private func drawHand(_ hand: HandLandmarks, in context: CGContext, width: Int, height: Int, colorIndex: Int) {
        let colors: [CGColor] = [
            CGColor(red: 0, green: 1, blue: 0, alpha: 1),    // Green - thumb
            CGColor(red: 0, green: 1, blue: 1, alpha: 1),    // Cyan - index
            CGColor(red: 0, green: 0.5, blue: 1, alpha: 1),  // Blue - middle
            CGColor(red: 1, green: 0.5, blue: 0.8, alpha: 1),// Pink - ring
            CGColor(red: 1, green: 0.3, blue: 0.3, alpha: 1),// Red - little
        ]

        let fingerChains: [[KeyPath<HandLandmarks, Point2D?>]] = [
            [\.wrist, \.thumbCMC, \.thumbMP, \.thumbIP, \.thumbTip],
            [\.wrist, \.indexMCP, \.indexPIP, \.indexDIP, \.indexTip],
            [\.wrist, \.middleMCP, \.middlePIP, \.middleDIP, \.middleTip],
            [\.wrist, \.ringMCP, \.ringPIP, \.ringDIP, \.ringTip],
            [\.wrist, \.littleMCP, \.littlePIP, \.littleDIP, \.littleTip],
        ]

        // Draw finger chains
        for (fingerIdx, chain) in fingerChains.enumerated() {
            let color = colors[fingerIdx % colors.count]
            context.setStrokeColor(color)
            context.setLineWidth(3)
            context.setLineCap(.round)

            var isFirst = true
            for keyPath in chain {
                guard let point = hand[keyPath: keyPath], point.confidence >= config.minConfidence else { continue }

                let x = CGFloat(point.x) * CGFloat(width)
                let y = CGFloat(1.0 - point.y) * CGFloat(height)  // Flip y: Vision uses bottom-left origin

                if isFirst {
                    context.move(to: CGPoint(x: x, y: y))
                    isFirst = false
                } else {
                    context.addLine(to: CGPoint(x: x, y: y))
                }
            }
            context.strokePath()
        }

        // Draw joints as circles
        let allJoints: [KeyPath<HandLandmarks, Point2D?>] = [
            \.wrist,
            \.thumbCMC, \.thumbMP, \.thumbIP, \.thumbTip,
            \.indexMCP, \.indexPIP, \.indexDIP, \.indexTip,
            \.middleMCP, \.middlePIP, \.middleDIP, \.middleTip,
            \.ringMCP, \.ringPIP, \.ringDIP, \.ringTip,
            \.littleMCP, \.littlePIP, \.littleDIP, \.littleTip,
        ]

        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        for keyPath in allJoints {
            guard let point = hand[keyPath: keyPath], point.confidence >= config.minConfidence else { continue }

            let x = CGFloat(point.x) * CGFloat(width)
            let y = CGFloat(1.0 - point.y) * CGFloat(height)  // Flip y: Vision uses bottom-left origin
            let radius: CGFloat = 4

            context.fillEllipse(in: CGRect(x: x - radius, y: y - radius, width: radius * 2, height: radius * 2))
        }
    }

    private func drawBody(_ body: BodyLandmarks, in context: CGContext, width: Int, height: Int) {
        let color = CGColor(red: 1, green: 1, blue: 0, alpha: 1)  // Yellow
        context.setStrokeColor(color)
        context.setLineWidth(2)
        context.setLineCap(.round)

        let connections: [(KeyPath<BodyLandmarks, Point2D?>, KeyPath<BodyLandmarks, Point2D?>)] = [
            // Head
            (\.leftEar, \.leftEye), (\.leftEye, \.nose), (\.nose, \.rightEye), (\.rightEye, \.rightEar),
            // Torso
            (\.leftShoulder, \.rightShoulder),
            (\.leftShoulder, \.leftHip), (\.rightShoulder, \.rightHip),
            (\.leftHip, \.rightHip),
            // Arms
            (\.leftShoulder, \.leftElbow), (\.leftElbow, \.leftWrist),
            (\.rightShoulder, \.rightElbow), (\.rightElbow, \.rightWrist),
            // Legs
            (\.leftHip, \.leftKnee), (\.leftKnee, \.leftAnkle),
            (\.rightHip, \.rightKnee), (\.rightKnee, \.rightAnkle),
        ]

        for (from, to) in connections {
            guard let p1 = body[keyPath: from], p1.confidence >= config.minConfidence,
                  let p2 = body[keyPath: to], p2.confidence >= config.minConfidence else { continue }

            let x1 = CGFloat(p1.x) * CGFloat(width)
            let y1 = CGFloat(1.0 - p1.y) * CGFloat(height)  // Flip y: Vision uses bottom-left origin
            let x2 = CGFloat(p2.x) * CGFloat(width)
            let y2 = CGFloat(1.0 - p2.y) * CGFloat(height)  // Flip y: Vision uses bottom-left origin

            context.move(to: CGPoint(x: x1, y: y1))
            context.addLine(to: CGPoint(x: x2, y: y2))
            context.strokePath()
        }

        // Draw joints
        context.setFillColor(color)
        let joints: [KeyPath<BodyLandmarks, Point2D?>] = [
            \.nose, \.leftEye, \.rightEye, \.leftEar, \.rightEar,
            \.leftShoulder, \.rightShoulder, \.leftElbow, \.rightElbow,
            \.leftWrist, \.rightWrist, \.leftHip, \.rightHip,
            \.leftKnee, \.rightKnee, \.leftAnkle, \.rightAnkle,
        ]

        for keyPath in joints {
            guard let point = body[keyPath: keyPath], point.confidence >= config.minConfidence else { continue }

            let x = CGFloat(point.x) * CGFloat(width)
            let y = CGFloat(1.0 - point.y) * CGFloat(height)  // Flip y: Vision uses bottom-left origin
            let radius: CGFloat = 5

            context.fillEllipse(in: CGRect(x: x - radius, y: y - radius, width: radius * 2, height: radius * 2))
        }
    }

    private func createPixelBuffer(from data: Data) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?

        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,  // Required for Metal texture access
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,  // IOSurface backing for Metal
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            config.width,
            config.height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let expectedBytesPerRow = config.width * 4  // BGRA

        if config.pixelFormat == "rgb24" {
            // Convert RGB24 to BGRA
            data.withUnsafeBytes { srcPtr in
                guard let src = srcPtr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                let dst = baseAddress.assumingMemoryBound(to: UInt8.self)

                for y in 0..<config.height {
                    for x in 0..<config.width {
                        let srcIdx = (y * config.width + x) * 3
                        let dstIdx = y * bytesPerRow + x * 4

                        dst[dstIdx + 0] = src[srcIdx + 2]  // B
                        dst[dstIdx + 1] = src[srcIdx + 1]  // G
                        dst[dstIdx + 2] = src[srcIdx + 0]  // R
                        dst[dstIdx + 3] = 255              // A
                    }
                }
            }
        } else if config.pixelFormat == "rgba" {
            // Convert RGBA to BGRA
            data.withUnsafeBytes { srcPtr in
                guard let src = srcPtr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                let dst = baseAddress.assumingMemoryBound(to: UInt8.self)

                for y in 0..<config.height {
                    for x in 0..<config.width {
                        let srcIdx = (y * config.width + x) * 4
                        let dstIdx = y * bytesPerRow + x * 4

                        dst[dstIdx + 0] = src[srcIdx + 2]  // B
                        dst[dstIdx + 1] = src[srcIdx + 1]  // G
                        dst[dstIdx + 2] = src[srcIdx + 0]  // R
                        dst[dstIdx + 3] = src[srcIdx + 3]  // A
                    }
                }
            }
        } else {
            // BGRA - direct copy (handle stride mismatch)
            data.withUnsafeBytes { srcPtr in
                guard let src = srcPtr.baseAddress else { return }
                if bytesPerRow == expectedBytesPerRow {
                    memcpy(baseAddress, src, data.count)
                } else {
                    // Copy row by row to handle stride
                    let dst = baseAddress.assumingMemoryBound(to: UInt8.self)
                    let srcBytes = src.assumingMemoryBound(to: UInt8.self)
                    for y in 0..<config.height {
                        memcpy(dst + y * bytesPerRow, srcBytes + y * expectedBytesPerRow, expectedBytesPerRow)
                    }
                }
            }
        }

        return buffer
    }

    private func extractFrameData(from pixelBuffer: CVPixelBuffer) -> Data {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return Data(count: config.frameSize)
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        if config.pixelFormat == "rgb24" {
            // Convert BGRA back to RGB24
            var output = Data(count: config.frameSize)
            output.withUnsafeMutableBytes { dstPtr in
                guard let dst = dstPtr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                let src = baseAddress.assumingMemoryBound(to: UInt8.self)

                for y in 0..<config.height {
                    for x in 0..<config.width {
                        let srcIdx = y * bytesPerRow + x * 4
                        let dstIdx = (y * config.width + x) * 3

                        dst[dstIdx + 0] = src[srcIdx + 2]  // R
                        dst[dstIdx + 1] = src[srcIdx + 1]  // G
                        dst[dstIdx + 2] = src[srcIdx + 0]  // B
                    }
                }
            }
            return output
        } else if config.pixelFormat == "rgba" {
            // Convert BGRA back to RGBA
            var output = Data(count: config.frameSize)
            output.withUnsafeMutableBytes { dstPtr in
                guard let dst = dstPtr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                let src = baseAddress.assumingMemoryBound(to: UInt8.self)

                for y in 0..<config.height {
                    for x in 0..<config.width {
                        let srcIdx = y * bytesPerRow + x * 4
                        let dstIdx = (y * config.width + x) * 4

                        dst[dstIdx + 0] = src[srcIdx + 2]  // R
                        dst[dstIdx + 1] = src[srcIdx + 1]  // G
                        dst[dstIdx + 2] = src[srcIdx + 0]  // B
                        dst[dstIdx + 3] = src[srcIdx + 3]  // A
                    }
                }
            }
            return output
        } else {
            // BGRA - direct copy (handle stride)
            let expectedBytesPerRow = config.width * 4
            if bytesPerRow == expectedBytesPerRow {
                return Data(bytes: baseAddress, count: config.frameSize)
            } else {
                var output = Data(count: config.frameSize)
                output.withUnsafeMutableBytes { dstPtr in
                    guard let dst = dstPtr.baseAddress else { return }
                    let src = baseAddress.assumingMemoryBound(to: UInt8.self)
                    for y in 0..<config.height {
                        memcpy(dst + y * expectedBytesPerRow, src + y * bytesPerRow, expectedBytesPerRow)
                    }
                }
                return output
            }
        }
    }
}
