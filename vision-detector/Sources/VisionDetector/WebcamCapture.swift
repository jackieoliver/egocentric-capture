import Foundation
import AVFoundation
import CoreMedia

/// Captures frames from the MacBook's built-in webcam using AVFoundation
final class WebcamCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "webcam.session")
    private let videoQueue = DispatchQueue(label: "webcam.frames", qos: .userInteractive)

    var onFrame: ((CVPixelBuffer, CMTime) -> Void)?

    func start() throws {
        try sessionQueue.sync {
            session.beginConfiguration()
            session.sessionPreset = .high

            // Get the default video device (built-in webcam)
            guard let device = AVCaptureDevice.default(for: .video) else {
                throw CaptureError.noCamera
            }

            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else {
                throw CaptureError.cannotAddInput
            }
            session.addInput(input)

            // Setup video output
            let output = AVCaptureVideoDataOutput()
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: videoQueue)

            guard session.canAddOutput(output) else {
                throw CaptureError.cannotAddOutput
            }
            session.addOutput(output)

            // Mirror the preview (more natural for self-view)
            if let connection = output.connection(with: .video) {
                if connection.isVideoMirroringSupported {
                    connection.isVideoMirrored = true
                }
            }

            session.commitConfiguration()
            session.startRunning()
        }
    }

    func stop() {
        sessionQueue.async {
            self.session.stopRunning()
        }
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        onFrame?(pixelBuffer, pts)
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Frame dropped due to late processing - this is expected under load
    }

    enum CaptureError: Error, LocalizedError {
        case noCamera
        case cannotAddInput
        case cannotAddOutput

        var errorDescription: String? {
            switch self {
            case .noCamera:
                return "No camera found. Make sure your MacBook has a built-in camera."
            case .cannotAddInput:
                return "Cannot add camera input to capture session"
            case .cannotAddOutput:
                return "Cannot add video output to capture session"
            }
        }
    }
}
