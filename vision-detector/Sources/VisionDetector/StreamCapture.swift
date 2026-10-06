import Foundation
import AVFoundation
import CoreMedia
import CoreVideo

/// Captures frames from an RTSP/UDP stream using AVFoundation
/// Falls back to FFmpeg subprocess if AVFoundation doesn't support the stream format
final class StreamCapture: NSObject {
    private let url: String
    private let cameraId: String
    private var asset: AVAsset?
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var ffmpegProcess: Process?
    private var ffmpegPipe: Pipe?

    private let captureQueue = DispatchQueue(label: "stream.capture", qos: .userInteractive)
    private var isRunning = false

    var onFrame: ((CVPixelBuffer, CMTime) -> Void)?
    var onError: ((Error) -> Void)?

    init(url: String, cameraId: String) {
        self.url = url
        self.cameraId = cameraId
        super.init()
    }

    /// Start capturing frames from the stream
    func start() {
        isRunning = true
        captureQueue.async { [weak self] in
            self?.startFFmpegCapture()
        }
    }

    /// Stop capturing
    func stop() {
        isRunning = false
        ffmpegProcess?.terminate()
        ffmpegProcess = nil
        ffmpegPipe = nil
    }

    // MARK: - FFmpeg-based capture (more reliable for RTSP/UDP)

    private func startFFmpegCapture() {
        let pipe = Pipe()
        self.ffmpegPipe = pipe

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg")

        // FFmpeg args: read from stream, output raw BGRA frames to stdout
        process.arguments = [
            "-hide_banner",
            "-loglevel", "error",
            "-rtsp_transport", "udp",  // Use UDP for RTSP
            "-i", url,
            "-f", "rawvideo",
            "-pix_fmt", "bgra",
            "-vsync", "0",
            "-"  // Output to stdout
        ]

        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            self.ffmpegProcess = process

            // Read frames from pipe
            readFramesFromPipe(pipe)

        } catch {
            // Try alternative ffmpeg path
            process.executableURL = URL(fileURLWithPath: "/usr/local/bin/ffmpeg")
            do {
                try process.run()
                self.ffmpegProcess = process
                readFramesFromPipe(pipe)
            } catch {
                onError?(StreamError.ffmpegNotFound)
            }
        }
    }

    private func readFramesFromPipe(_ pipe: Pipe) {
        // We need to know frame dimensions - default to 1920x1080,
        // but this should be configurable or probed
        let width = 1920
        let height = 1080
        let bytesPerPixel = 4  // BGRA
        let frameSize = width * height * bytesPerPixel

        var frameCount: Int64 = 0
        let fps: Double = 30.0  // Assumed, should match stream

        captureQueue.async { [weak self] in
            guard let self = self else { return }

            let handle = pipe.fileHandleForReading
            var buffer = Data(capacity: frameSize)

            while self.isRunning {
                autoreleasepool {
                    let chunk = handle.readData(ofLength: frameSize - buffer.count)
                    if chunk.isEmpty {
                        // Stream ended or error
                        return
                    }

                    buffer.append(chunk)

                    // Process complete frames
                    while buffer.count >= frameSize {
                        let frameData = buffer.prefix(frameSize)
                        buffer = Data(buffer.dropFirst(frameSize))

                        // Create CVPixelBuffer from raw data
                        if let pixelBuffer = self.createPixelBuffer(
                            from: frameData,
                            width: width,
                            height: height
                        ) {
                            let pts = CMTime(
                                value: frameCount,
                                timescale: CMTimeScale(fps)
                            )
                            frameCount += 1

                            self.onFrame?(pixelBuffer, pts)
                        }
                    }
                }
            }
        }
    }

    private func createPixelBuffer(from data: Data, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?

        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true
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

        data.withUnsafeBytes { ptr in
            if let bytes = ptr.baseAddress {
                memcpy(baseAddress, bytes, data.count)
            }
        }

        return buffer
    }

    enum StreamError: Error, LocalizedError {
        case ffmpegNotFound
        case streamOpenFailed
        case invalidFrameData

        var errorDescription: String? {
            switch self {
            case .ffmpegNotFound:
                return "FFmpeg not found. Install with: brew install ffmpeg"
            case .streamOpenFailed:
                return "Failed to open video stream"
            case .invalidFrameData:
                return "Invalid frame data received"
            }
        }
    }
}
