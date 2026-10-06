import Foundation
import Vision
import CoreVideo
import CoreMedia
import QuartzCore  // For CACurrentMediaTime

/// Processes CVPixelBuffers through Vision framework for hand and body pose detection
final class VisionProcessor {
    private let config: DetectorConfig
    private let handRequest: VNDetectHumanHandPoseRequest
    private let bodyRequest: VNDetectHumanBodyPoseRequest
    private let visionQueue = DispatchQueue(label: "vision.processing", qos: .userInteractive)

    private var lastProcessTime: CFTimeInterval = 0
    private var frameNumber: Int = 0
    private let startTime: CFTimeInterval

    var onDetection: ((DetectionFrame) -> Void)?

    init(config: DetectorConfig = .default) {
        self.config = config
        self.startTime = CACurrentMediaTime()

        self.handRequest = VNDetectHumanHandPoseRequest()
        self.handRequest.maximumHandCount = config.maxHands

        self.bodyRequest = VNDetectHumanBodyPoseRequest()
    }

    /// Process a pixel buffer and emit detections
    /// - Parameters:
    ///   - pixelBuffer: The frame to process
    ///   - cameraId: Identifier for the source camera
    ///   - presentationTime: Optional timestamp from the stream
    func process(_ pixelBuffer: CVPixelBuffer, cameraId: String, presentationTime: CMTime? = nil) {
        let now = CACurrentMediaTime()
        let minInterval = 1.0 / config.targetFPS

        // Throttle to target FPS
        guard now - lastProcessTime >= minInterval else { return }
        lastProcessTime = now

        let currentFrame = frameNumber
        frameNumber += 1

        visionQueue.async { [weak self] in
            guard let self = self else { return }

            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: .up,
                options: [:]
            )

            var requests: [VNRequest] = []
            if self.config.detectHands {
                requests.append(self.handRequest)
            }
            if self.config.detectBody {
                requests.append(self.bodyRequest)
            }

            do {
                try handler.perform(requests)

                // Extract hand landmarks
                var hands: [HandLandmarks] = []
                if self.config.detectHands,
                   let handObservations = self.handRequest.results {
                    for observation in handObservations {
                        if let landmarks = try? HandLandmarks(from: observation) {
                            hands.append(landmarks)
                        }
                    }
                }

                // Extract body landmarks
                var body: BodyLandmarks? = nil
                if self.config.detectBody,
                   let bodyObservations = self.bodyRequest.results,
                   let firstBody = bodyObservations.first {
                    body = try? BodyLandmarks(from: firstBody)
                }

                // Calculate timestamp
                let timestamp: Double
                if let pt = presentationTime, pt.isValid {
                    timestamp = CMTimeGetSeconds(pt)
                } else {
                    timestamp = now - self.startTime
                }

                let frame = DetectionFrame(
                    timestamp: timestamp,
                    frameNumber: currentFrame,
                    sourceCamera: cameraId,
                    hands: hands,
                    body: body
                )

                self.onDetection?(frame)

            } catch {
                // Skip frame on error, don't crash
                fputs("Warning: Vision processing failed: \(error.localizedDescription)\n", stderr)
            }
        }
    }
}
