import Foundation
import Vision

// MARK: - Output Models (JSON serializable)

struct Point2D: Codable {
    let x: Double
    let y: Double
    let confidence: Float

    init(from point: VNRecognizedPoint) {
        self.x = point.location.x
        self.y = point.location.y
        self.confidence = point.confidence
    }

    init(x: Double, y: Double, confidence: Float) {
        self.x = x
        self.y = y
        self.confidence = confidence
    }
}

struct HandLandmarks: Codable {
    let chirality: String  // "left" or "right"
    let avgConfidence: Float  // Average confidence across all landmarks
    let wrist: Point2D?
    let thumbCMC: Point2D?
    let thumbMP: Point2D?
    let thumbIP: Point2D?
    let thumbTip: Point2D?
    let indexMCP: Point2D?
    let indexPIP: Point2D?
    let indexDIP: Point2D?
    let indexTip: Point2D?
    let middleMCP: Point2D?
    let middlePIP: Point2D?
    let middleDIP: Point2D?
    let middleTip: Point2D?
    let ringMCP: Point2D?
    let ringPIP: Point2D?
    let ringDIP: Point2D?
    let ringTip: Point2D?
    let littleMCP: Point2D?
    let littlePIP: Point2D?
    let littleDIP: Point2D?
    let littleTip: Point2D?

    init(from observation: VNHumanHandPoseObservation) throws {
        self.chirality = observation.chirality == .left ? "left" :
                         observation.chirality == .right ? "right" : "unknown"

        let points = try observation.recognizedPoints(.all)

        self.wrist = points[.wrist].map { Point2D(from: $0) }
        self.thumbCMC = points[.thumbCMC].map { Point2D(from: $0) }
        self.thumbMP = points[.thumbMP].map { Point2D(from: $0) }
        self.thumbIP = points[.thumbIP].map { Point2D(from: $0) }
        self.thumbTip = points[.thumbTip].map { Point2D(from: $0) }
        self.indexMCP = points[.indexMCP].map { Point2D(from: $0) }
        self.indexPIP = points[.indexPIP].map { Point2D(from: $0) }
        self.indexDIP = points[.indexDIP].map { Point2D(from: $0) }
        self.indexTip = points[.indexTip].map { Point2D(from: $0) }
        self.middleMCP = points[.middleMCP].map { Point2D(from: $0) }
        self.middlePIP = points[.middlePIP].map { Point2D(from: $0) }
        self.middleDIP = points[.middleDIP].map { Point2D(from: $0) }
        self.middleTip = points[.middleTip].map { Point2D(from: $0) }
        self.ringMCP = points[.ringMCP].map { Point2D(from: $0) }
        self.ringPIP = points[.ringPIP].map { Point2D(from: $0) }
        self.ringDIP = points[.ringDIP].map { Point2D(from: $0) }
        self.ringTip = points[.ringTip].map { Point2D(from: $0) }
        self.littleMCP = points[.littleMCP].map { Point2D(from: $0) }
        self.littlePIP = points[.littlePIP].map { Point2D(from: $0) }
        self.littleDIP = points[.littleDIP].map { Point2D(from: $0) }
        self.littleTip = points[.littleTip].map { Point2D(from: $0) }

        // Calculate average confidence across all landmarks
        let allPoints: [Point2D?] = [
            self.wrist, self.thumbCMC, self.thumbMP, self.thumbIP, self.thumbTip,
            self.indexMCP, self.indexPIP, self.indexDIP, self.indexTip,
            self.middleMCP, self.middlePIP, self.middleDIP, self.middleTip,
            self.ringMCP, self.ringPIP, self.ringDIP, self.ringTip,
            self.littleMCP, self.littlePIP, self.littleDIP, self.littleTip
        ]
        let confidences = allPoints.compactMap { $0?.confidence }
        self.avgConfidence = confidences.isEmpty ? 0 : confidences.reduce(0, +) / Float(confidences.count)
    }
}

struct BodyLandmarks: Codable {
    // Head
    let nose: Point2D?
    let leftEye: Point2D?
    let rightEye: Point2D?
    let leftEar: Point2D?
    let rightEar: Point2D?

    // Torso
    let neck: Point2D?
    let leftShoulder: Point2D?
    let rightShoulder: Point2D?
    let leftHip: Point2D?
    let rightHip: Point2D?
    let root: Point2D?  // center hip

    // Left arm
    let leftElbow: Point2D?
    let leftWrist: Point2D?

    // Right arm
    let rightElbow: Point2D?
    let rightWrist: Point2D?

    // Left leg
    let leftKnee: Point2D?
    let leftAnkle: Point2D?

    // Right leg
    let rightKnee: Point2D?
    let rightAnkle: Point2D?

    init(from observation: VNHumanBodyPoseObservation) throws {
        let points = try observation.recognizedPoints(.all)

        self.nose = points[.nose].map { Point2D(from: $0) }
        self.leftEye = points[.leftEye].map { Point2D(from: $0) }
        self.rightEye = points[.rightEye].map { Point2D(from: $0) }
        self.leftEar = points[.leftEar].map { Point2D(from: $0) }
        self.rightEar = points[.rightEar].map { Point2D(from: $0) }
        self.neck = points[.neck].map { Point2D(from: $0) }
        self.leftShoulder = points[.leftShoulder].map { Point2D(from: $0) }
        self.rightShoulder = points[.rightShoulder].map { Point2D(from: $0) }
        self.leftHip = points[.leftHip].map { Point2D(from: $0) }
        self.rightHip = points[.rightHip].map { Point2D(from: $0) }
        self.root = points[.root].map { Point2D(from: $0) }
        self.leftElbow = points[.leftElbow].map { Point2D(from: $0) }
        self.leftWrist = points[.leftWrist].map { Point2D(from: $0) }
        self.rightElbow = points[.rightElbow].map { Point2D(from: $0) }
        self.rightWrist = points[.rightWrist].map { Point2D(from: $0) }
        self.leftKnee = points[.leftKnee].map { Point2D(from: $0) }
        self.leftAnkle = points[.leftAnkle].map { Point2D(from: $0) }
        self.rightKnee = points[.rightKnee].map { Point2D(from: $0) }
        self.rightAnkle = points[.rightAnkle].map { Point2D(from: $0) }
    }
}

struct DetectionFrame: Codable {
    let timestamp: Double           // seconds since start
    let frameNumber: Int
    let sourceCamera: String        // camera identifier (e.g., "head", "rwrist")
    let hands: [HandLandmarks]
    let body: BodyLandmarks?

    func toJSONLine() -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = []  // compact, single line
        guard let data = try? encoder.encode(self) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

// MARK: - Stats Output Models (JSON serializable)

struct StatsFrame: Codable {
    let timestamp: Double           // seconds since start
    let frameNumber: Int
    let sourceCamera: String

    // Latency stats (milliseconds)
    let latencyP50: Double?
    let latencyP99: Double?
    let latencyLatest: Double?

    // FPS stats
    let fpsP50: Double?
    let fpsLatest: Double?

    // Confidence stats (0.0 - 1.0)
    let confidenceP50: Double?
    let confidenceP1: Double?
    let confidenceLatest: Double?

    // Hand presence (0.0 - 1.0 rolling average)
    let leftHandPresence: Double?
    let rightHandPresence: Double?
    let bimanualPresence: Double?

    // Counters
    let totalFrames: Int
    let framesWithHands: Int
    let framesBimanual: Int
    let runtime: Double

    func toJSONLine() -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = []  // compact, single line
        guard let data = try? encoder.encode(self) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

// MARK: - Configuration

struct DetectorConfig {
    let maxHands: Int
    let minConfidence: Float
    let targetFPS: Double
    let detectBody: Bool
    let detectHands: Bool

    static let `default` = DetectorConfig(
        maxHands: 2,
        minConfidence: 0.3,
        targetFPS: 30.0,
        detectBody: true,
        detectHands: true
    )
}
