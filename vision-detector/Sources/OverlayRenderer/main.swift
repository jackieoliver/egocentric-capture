import SwiftUI
import ArgumentParser
import AVFoundation

@main
struct OverlayRendererApp: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "overlay-renderer",
        abstract: "Real-time skeleton overlay renderer for hand and body pose visualization",
        version: "0.1.0"
    )

    @Option(name: .shortAndLong, help: "Stream URL (RTSP/UDP). Can specify multiple for tiled view.")
    var stream: [String] = []

    @Option(name: .shortAndLong, help: "Camera identifier for each stream (in order)")
    var camera: [String] = []

    @Option(name: .long, help: "Window width")
    var width: Int = 1280

    @Option(name: .long, help: "Window height")
    var height: Int = 720

    @Flag(name: .long, inversion: .prefixedNo, help: "Read landmarks from stdin (JSONL format)")
    var stdin: Bool = true

    @Option(name: .long, help: "Record output to file (MP4)")
    var record: String?

    func run() throws {
        // Build camera configs
        var cameras: [CameraConfig] = []
        for (index, url) in stream.enumerated() {
            let id = index < camera.count ? camera[index] : "camera\(index)"
            cameras.append(CameraConfig(id: id, url: url))
        }

        // If no streams provided, show a placeholder
        if cameras.isEmpty {
            fputs("No streams provided. Use --stream to specify RTSP/UDP URLs.\n", stderr)
            fputs("Example: overlay-renderer --stream udp://@:8554 --camera head\n", stderr)
            fputs("\nStarting with demo mode (no video, landmarks only from stdin)...\n", stderr)
        }

        // Capture values for the closure
        let windowWidth = CGFloat(width)
        let windowHeight = CGFloat(height)
        let readFromStdin = stdin
        let recordPath = record

        // Start the app on main thread
        DispatchQueue.main.async {
            let app = NSApplication.shared
            app.setActivationPolicy(.regular)

            let appState = AppState(
                cameras: cameras,
                windowSize: CGSize(width: windowWidth, height: windowHeight),
                readFromStdin: readFromStdin,
                recordPath: recordPath
            )

            let delegate = AppDelegate(state: appState)
            app.delegate = delegate
            app.run()
        }

        // Keep running
        RunLoop.current.run()
    }
}

// MARK: - App State

struct CameraConfig: Identifiable, Sendable {
    let id: String
    let url: String
}

@MainActor
class AppState: ObservableObject {
    let cameras: [CameraConfig]
    let windowSize: CGSize
    let readFromStdin: Bool
    let recordPath: String?

    @Published var landmarks: [String: DetectionData] = [:]  // camera_id -> latest detection

    init(cameras: [CameraConfig], windowSize: CGSize, readFromStdin: Bool, recordPath: String?) {
        self.cameras = cameras
        self.windowSize = windowSize
        self.readFromStdin = readFromStdin
        self.recordPath = recordPath
    }

    func updateLandmarks(_ data: DetectionData) {
        landmarks[data.sourceCamera] = data
    }
}

// MARK: - App Delegate

@MainActor
class AppDelegate: NSObject, NSApplicationDelegate {
    let state: AppState
    var window: NSWindow?
    var landmarkReceiver: LandmarkReceiver?

    init(state: AppState) {
        self.state = state
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create window
        let contentView = MainContentView(state: state)

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: state.windowSize.width, height: state.windowSize.height),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )

        window?.title = "Pose Overlay"
        window?.contentView = NSHostingView(rootView: contentView)
        window?.center()
        window?.makeKeyAndOrderFront(nil)

        // Start reading landmarks from stdin
        if state.readFromStdin {
            landmarkReceiver = LandmarkReceiver { [weak self] data in
                Task { @MainActor in
                    self?.state.updateLandmarks(data)
                }
            }
            landmarkReceiver?.start()
        }

        NSApp.activate(ignoringOtherApps: true)
    }

    nonisolated func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// MARK: - Main Content View

struct MainContentView: View {
    @ObservedObject var state: AppState

    var body: some View {
        GeometryReader { geometry in
            if state.cameras.isEmpty {
                // Demo mode - just show skeleton on black background
                ZStack {
                    Color.black

                    ForEach(Array(state.landmarks.values), id: \.sourceCamera) { data in
                        SkeletonOverlayView(detection: data, frameSize: geometry.size)
                    }

                    if state.landmarks.isEmpty {
                        VStack {
                            Text("Waiting for landmarks on stdin...")
                                .foregroundColor(.gray)
                            Text("Pipe JSONL from VisionDetector")
                                .font(.caption)
                                .foregroundColor(.gray.opacity(0.7))
                        }
                    }
                }
            } else if state.cameras.count == 1 {
                // Single camera view
                SingleCameraView(
                    config: state.cameras[0],
                    detection: state.landmarks[state.cameras[0].id],
                    frameSize: geometry.size
                )
            } else {
                // Multi-camera tiled view
                MultiCameraView(
                    cameras: state.cameras,
                    landmarks: state.landmarks,
                    frameSize: geometry.size
                )
            }
        }
    }
}
