import SwiftUI
import AVFoundation
import AVKit

/// Single camera view with video preview and skeleton overlay
struct SingleCameraView: View {
    let config: CameraConfig
    let detection: DetectionData?
    let frameSize: CGSize

    var body: some View {
        ZStack {
            // Video layer
            VideoPlayerView(url: config.url)

            // Skeleton overlay
            if let detection = detection {
                SkeletonOverlayView(detection: detection, frameSize: frameSize)
            }

            // Camera label
            VStack {
                HStack {
                    Text(config.id.uppercased())
                        .font(.system(size: 14, weight: .bold, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(4)
                    Spacer()
                }
                Spacer()
            }
            .padding(8)
        }
    }
}

/// AVPlayer-based video view for RTSP/UDP streams
struct VideoPlayerView: NSViewRepresentable {
    let url: String

    func makeNSView(context: Context) -> AVPlayerView {
        let playerView = AVPlayerView()
        playerView.controlsStyle = .none
        playerView.videoGravity = .resizeAspectFill

        // Create player for the stream URL
        if let streamURL = URL(string: url) {
            let player = AVPlayer(url: streamURL)
            player.isMuted = true
            playerView.player = player
            player.play()
        }

        return playerView
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        // URL changes are not expected during runtime
    }
}

/// Alternative: FFmpeg-based video capture for formats AVPlayer doesn't support
/// Uses MTKView for Metal-accelerated rendering
class FFmpegStreamCapture: NSObject {
    private let url: String
    private var process: Process?
    private var pipe: Pipe?
    private let captureQueue = DispatchQueue(label: "ffmpeg.capture", qos: .userInteractive)

    var onFrame: ((CVPixelBuffer) -> Void)?

    init(url: String) {
        self.url = url
        super.init()
    }

    func start(width: Int = 1920, height: Int = 1080) {
        captureQueue.async { [weak self] in
            self?.startFFmpeg(width: width, height: height)
        }
    }

    func stop() {
        process?.terminate()
        process = nil
        pipe = nil
    }

    private func startFFmpeg(width: Int, height: Int) {
        let pipe = Pipe()
        self.pipe = pipe

        let process = Process()

        // Try homebrew path first, then standard path
        let ffmpegPaths = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]
        var ffmpegPath: String?
        for path in ffmpegPaths {
            if FileManager.default.fileExists(atPath: path) {
                ffmpegPath = path
                break
            }
        }

        guard let path = ffmpegPath else {
            fputs("FFmpeg not found. Install with: brew install ffmpeg\n", stderr)
            return
        }

        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = [
            "-hide_banner",
            "-loglevel", "error",
            "-rtsp_transport", "udp",
            "-i", url,
            "-f", "rawvideo",
            "-pix_fmt", "bgra",
            "-s", "\(width)x\(height)",
            "-vsync", "0",
            "-"
        ]

        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            self.process = process
            readFrames(pipe: pipe, width: width, height: height)
        } catch {
            fputs("Failed to start FFmpeg: \(error.localizedDescription)\n", stderr)
        }
    }

    private func readFrames(pipe: Pipe, width: Int, height: Int) {
        let bytesPerPixel = 4
        let frameSize = width * height * bytesPerPixel
        let handle = pipe.fileHandleForReading
        var buffer = Data(capacity: frameSize)

        while process?.isRunning == true {
            autoreleasepool {
                let chunk = handle.readData(ofLength: frameSize - buffer.count)
                if chunk.isEmpty { return }

                buffer.append(chunk)

                while buffer.count >= frameSize {
                    let frameData = buffer.prefix(frameSize)
                    buffer = Data(buffer.dropFirst(frameSize))

                    if let pixelBuffer = createPixelBuffer(from: frameData, width: width, height: height) {
                        onFrame?(pixelBuffer)
                    }
                }
            }
        }
    }

    private func createPixelBuffer(from data: Data, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?

        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        data.withUnsafeBytes { ptr in
            if let bytes = ptr.baseAddress {
                memcpy(baseAddress, bytes, data.count)
            }
        }

        return buffer
    }
}
