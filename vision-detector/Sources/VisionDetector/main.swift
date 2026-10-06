import Foundation
import ArgumentParser

@main
struct VisionDetectorCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vision-detector",
        abstract: "Hand and body pose detection with optional overlay rendering",
        version: "0.2.0",
        subcommands: [Transform.self, Detect.self, Test.self],
        defaultSubcommand: Transform.self
    )
}

// MARK: - Transform Command (main pipeline mode)

struct Transform: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Process video frames: stdin → detection + overlay → stdout"
    )

    @Option(name: .shortAndLong, help: "Frame width in pixels")
    var width: Int = 1920

    @Option(name: .shortAndLong, help: "Frame height in pixels")
    var height: Int = 1080

    @Option(name: .long, help: "Input pixel format (bgra, rgba, rgb24)")
    var pixelFormat: String = "bgra"

    @Option(name: .long, help: "Target processing FPS (0 = unlimited)")
    var fps: Double = 30.0

    @Option(name: .long, help: "Maximum hands to detect (1-4)")
    var maxHands: Int = 2

    @Option(name: .long, help: "Minimum confidence threshold")
    var minConfidence: Float = 0.3

    @Option(name: .long, help: "Camera identifier for landmark output")
    var camera: String = "camera0"

    @Flag(name: .long, help: "Disable hand detection")
    var noHands: Bool = false

    @Flag(name: .long, help: "Disable body detection")
    var noBody: Bool = false

    @Flag(name: .long, help: "Disable overlay rendering (landmarks only to stderr)")
    var noOverlay: Bool = false

    @Flag(name: .long, help: "Use Metal GPU acceleration for overlay rendering")
    var metal: Bool = false

    @Flag(name: .long, help: "Async detection (non-blocking, uses stale results for lower latency)")
    var async: Bool = false

    @Flag(name: .long, help: "Enable stats collection and show live P50/P99 stats overlay")
    var stats: Bool = false

    @Flag(name: .long, help: "Disable stats overlay rendering (stats still collected if --stats is set)")
    var noStatsOverlay: Bool = false

    @Option(name: .shortAndLong, help: "Write landmarks to file instead of stderr")
    var landmarksOut: String?

    @Option(name: .long, help: "Write stats to file (JSONL format)")
    var statsOut: String?

    func validate() throws {
        guard width > 0 && width <= 7680 else {
            throw ValidationError("Width must be between 1 and 7680")
        }
        guard height > 0 && height <= 4320 else {
            throw ValidationError("Height must be between 1 and 4320")
        }
        guard maxHands >= 1 && maxHands <= 4 else {
            throw ValidationError("maxHands must be between 1 and 4")
        }
        guard ["bgra", "rgba", "rgb24"].contains(pixelFormat) else {
            throw ValidationError("pixelFormat must be bgra, rgba, or rgb24")
        }
    }

    func run() throws {
        fputs("Starting video processor...\n", stderr)
        fputs("Input: \(width)x\(height) \(pixelFormat)\n", stderr)
        fputs("Detection: hands=\(!noHands), body=\(!noBody), maxHands=\(maxHands), async=\(async)\n", stderr)
        fputs("Overlay: \(!noOverlay), Metal: \(metal), Stats: \(stats), StatsOverlay: \(!noStatsOverlay)\n", stderr)

        let config = ProcessorConfig(
            width: width,
            height: height,
            pixelFormat: pixelFormat,
            fps: fps,
            maxHands: maxHands,
            minConfidence: minConfidence,
            cameraId: camera,
            detectHands: !noHands,
            detectBody: !noBody,
            renderOverlay: !noOverlay,
            useMetal: metal,
            asyncDetection: async,
            showStats: stats,
            renderStatsOverlay: stats && !noStatsOverlay,
            landmarksOutPath: landmarksOut,
            statsOutPath: statsOut
        )

        let processor = FramePipelineProcessor(config: config)

        // Handle SIGINT gracefully
        let signalSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        signal(SIGINT, SIG_IGN)
        signalSource.setEventHandler {
            fputs("\nShutting down...\n", stderr)
            processor.stop()
            Darwin.exit(0)
        }
        signalSource.resume()

        // Run the processor (blocks until stdin closes)
        processor.run()
    }
}

// MARK: - Detect Command (landmarks only, for backward compatibility)

struct Detect: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run pose detection on video streams, output landmarks only (no overlay)"
    )

    @Option(name: .shortAndLong, help: "Stream URL (RTSP/UDP). Can specify multiple.")
    var stream: [String] = []

    @Option(name: .shortAndLong, help: "Camera identifier for each stream (in order)")
    var camera: [String] = []

    @Option(name: .long, help: "Maximum number of hands to detect (1-4)")
    var maxHands: Int = 2

    @Option(name: .long, help: "Target processing FPS")
    var fps: Double = 30.0

    @Option(name: .long, help: "Minimum confidence threshold (0.0-1.0)")
    var minConfidence: Float = 0.3

    @Flag(name: .long, help: "Disable hand detection")
    var noHands: Bool = false

    @Flag(name: .long, help: "Disable body detection")
    var noBody: Bool = false

    @Option(name: .shortAndLong, help: "Output file path (default: stdout)")
    var output: String?

    func validate() throws {
        guard !stream.isEmpty else {
            throw ValidationError("At least one stream URL is required")
        }
    }

    func run() throws {
        // Setup output
        let outputHandle: FileHandle
        if let outputPath = output {
            FileManager.default.createFile(atPath: outputPath, contents: nil)
            outputHandle = FileHandle(forWritingAtPath: outputPath)!
        } else {
            outputHandle = FileHandle.standardOutput
        }

        fputs("Starting detector (landmarks only)...\n", stderr)
        fputs("Streams: \(stream.joined(separator: ", "))\n", stderr)

        let config = DetectorConfig(
            maxHands: maxHands,
            minConfidence: minConfidence,
            targetFPS: fps,
            detectBody: !noBody,
            detectHands: !noHands
        )

        let processor = VisionProcessor(config: config)

        let outputLock = NSLock()
        processor.onDetection = { frame in
            if let jsonLine = frame.toJSONLine() {
                outputLock.lock()
                outputHandle.write((jsonLine + "\n").data(using: .utf8)!)
                outputLock.unlock()
            }
        }

        // Start captures for each stream
        var captures: [StreamCapture] = []

        for (index, url) in stream.enumerated() {
            let cameraId = index < camera.count ? camera[index] : "camera\(index)"

            let capture = StreamCapture(url: url, cameraId: cameraId)
            capture.onFrame = { pixelBuffer, pts in
                processor.process(pixelBuffer, cameraId: cameraId, presentationTime: pts)
            }
            capture.onError = { error in
                fputs("Error on \(cameraId): \(error.localizedDescription)\n", stderr)
            }

            capture.start()
            captures.append(capture)
            fputs("Started capture for \(cameraId) from \(url)\n", stderr)
        }

        fputs("Detection running. Press Ctrl+C to stop.\n", stderr)

        // Keep running
        let runLoop = RunLoop.current
        let signalSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        signal(SIGINT, SIG_IGN)

        signalSource.setEventHandler {
            fputs("\nStopping...\n", stderr)
            for capture in captures {
                capture.stop()
            }
            CFRunLoopStop(runLoop.getCFRunLoop())
        }
        signalSource.resume()

        runLoop.run()
    }
}

// MARK: - Test Command (webcam)

struct Test: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Test pose detection using the built-in webcam"
    )

    @Option(name: .long, help: "Maximum number of hands to detect")
    var maxHands: Int = 2

    @Option(name: .long, help: "Target processing FPS")
    var fps: Double = 30.0

    func run() throws {
        fputs("Starting webcam test mode...\n", stderr)
        fputs("This will use your MacBook's built-in camera.\n", stderr)

        let config = DetectorConfig(
            maxHands: maxHands,
            minConfidence: 0.3,
            targetFPS: fps,
            detectBody: true,
            detectHands: true
        )

        let processor = VisionProcessor(config: config)
        processor.onDetection = { frame in
            if let jsonLine = frame.toJSONLine() {
                print(jsonLine)
            }
        }

        let webcam = WebcamCapture()
        webcam.onFrame = { pixelBuffer, pts in
            processor.process(pixelBuffer, cameraId: "webcam", presentationTime: pts)
        }

        do {
            try webcam.start()
            fputs("Webcam started. Press Ctrl+C to stop.\n", stderr)

            let runLoop = RunLoop.current
            let signalSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
            signal(SIGINT, SIG_IGN)

            signalSource.setEventHandler {
                fputs("\nStopping...\n", stderr)
                webcam.stop()
                CFRunLoopStop(runLoop.getCFRunLoop())
            }
            signalSource.resume()

            runLoop.run()

        } catch {
            fputs("Failed to start webcam: \(error.localizedDescription)\n", stderr)
            throw ExitCode.failure
        }
    }
}
