import Foundation
import CoreGraphics
import CoreVideo
import QuartzCore
import CoreText

// MARK: - Color Palette

struct StatsColors {
    static let bg = CGColor(red: 25/255, green: 25/255, blue: 25/255, alpha: 1)
    static let bgLight = CGColor(red: 35/255, green: 35/255, blue: 35/255, alpha: 1)
    static let border = CGColor(red: 60/255, green: 60/255, blue: 60/255, alpha: 1)
    static let text = CGColor(red: 220/255, green: 220/255, blue: 220/255, alpha: 1)
    static let textDim = CGColor(red: 140/255, green: 140/255, blue: 140/255, alpha: 1)
    static let textLabel = CGColor(red: 180/255, green: 180/255, blue: 180/255, alpha: 1)
    static let accentCyan = CGColor(red: 0/255, green: 200/255, blue: 230/255, alpha: 1)      // Latency
    static let accentGreen = CGColor(red: 100/255, green: 220/255, blue: 100/255, alpha: 1)  // FPS
    static let accentOrange = CGColor(red: 255/255, green: 180/255, blue: 80/255, alpha: 1)  // Confidence
    static let accentBlue = CGColor(red: 100/255, green: 180/255, blue: 255/255, alpha: 1)   // Hand presence
}

// MARK: - Rolling Statistics

struct RollingStats {
    private var values: [Double] = []
    private let windowSize: Int

    init(windowSize: Int = 300) {
        self.windowSize = windowSize
    }

    mutating func add(_ value: Double) {
        values.append(value)
        if values.count > windowSize {
            values.removeFirst()
        }
    }

    func percentile(_ p: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let index = Int(Double(sorted.count - 1) * p / 100.0)
        return sorted[min(index, sorted.count - 1)]
    }

    var p50: Double? { percentile(50) }
    var p99: Double? { percentile(99) }
    var p1: Double? { percentile(1) }

    var mean: Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    var latest: Double? {
        values.last
    }

    var allValues: [Double] {
        values
    }

    var count: Int {
        values.count
    }

    mutating func clear() {
        values.removeAll()
    }
}

// MARK: - Timegraph Renderer

struct Timegraph {
    let width: Int
    let height: Int
    let lineColor: CGColor
    let fill: Bool
    let lineWidth: CGFloat

    init(
        width: Int = 160,
        height: Int = 40,
        lineColor: CGColor = StatsColors.accentGreen,
        fill: Bool = true,
        lineWidth: CGFloat = 2
    ) {
        self.width = width
        self.height = height
        self.lineColor = lineColor
        self.fill = fill
        self.lineWidth = lineWidth
    }

    func render(values: [Double], minVal: Double? = nil, maxVal: Double? = nil, context: CGContext, x: Int, y: Int) {
        let rect = CGRect(x: x, y: y, width: width, height: height)

        // Background
        context.setFillColor(StatsColors.bgLight)
        context.fill(rect)

        guard values.count >= 2 else { return }

        // Use percentile-based bounds to handle outliers (ignore extreme spikes)
        let sorted = values.sorted()
        let p5Index = max(0, Int(Double(sorted.count) * 0.05))
        let p95Index = min(sorted.count - 1, Int(Double(sorted.count) * 0.95))
        let dataMin = sorted[p5Index]
        let dataMax = sorted[p95Index]

        var actualMin = minVal ?? (dataMin * 0.9)
        var actualMax = maxVal ?? (dataMax * 1.1)

        var valRange = actualMax - actualMin
        if valRange < 0.001 {
            valRange = 1.0
            actualMin = dataMin - 0.5
            actualMax = dataMax + 0.5
        }

        // Take only the most recent 'width' samples (1 sample per pixel)
        let sampled: [Double]
        if values.count > width {
            sampled = Array(values.suffix(width))
        } else {
            sampled = values
        }

        // Convert to points - align to RIGHT edge of graph
        let paddingY: CGFloat = 4
        let usableHeight = CGFloat(height) - paddingY * 2
        let startX = x + width - sampled.count  // Right-align the data

        var points: [CGPoint] = []
        for (i, v) in sampled.enumerated() {
            let px = CGFloat(startX + i)
            // Clamp normalized value to [0, 1] to handle outliers gracefully
            let normalized = max(0, min(1, (v - actualMin) / valRange))
            let py = CGFloat(y) + paddingY + (1 - CGFloat(normalized)) * usableHeight
            points.append(CGPoint(x: px, y: py))
        }

        // Draw filled area
        if fill && points.count >= 2 {
            let path = CGMutablePath()
            path.move(to: CGPoint(x: points[0].x, y: CGFloat(y + height)))
            for point in points {
                path.addLine(to: point)
            }
            path.addLine(to: CGPoint(x: points.last!.x, y: CGFloat(y + height)))
            path.closeSubpath()

            context.saveGState()
            context.setFillColor(lineColor.copy(alpha: 0.25)!)
            context.addPath(path)
            context.fillPath()
            context.restoreGState()
        }

        // Draw line
        if points.count >= 2 {
            let path = CGMutablePath()
            path.move(to: points[0])
            for i in 1..<points.count {
                path.addLine(to: points[i])
            }

            context.setStrokeColor(lineColor)
            context.setLineWidth(lineWidth)
            context.addPath(path)
            context.strokePath()
        }

        // Draw baseline
        context.setStrokeColor(StatsColors.border)
        context.setLineWidth(1)
        context.move(to: CGPoint(x: x, y: y + height - 1))
        context.addLine(to: CGPoint(x: x + width, y: y + height - 1))
        context.strokePath()
    }
}

// MARK: - Pipeline Stats

final class PipelineStats {
    // Timing stats
    var latency = RollingStats(windowSize: 300)
    var fpsHistory = RollingStats(windowSize: 300)

    // Detection quality
    var handConfidence = RollingStats(windowSize: 300)
    var bodyConfidence = RollingStats(windowSize: 300)

    // Hand activity
    var leftHandPresent = RollingStats(windowSize: 300)
    var rightHandPresent = RollingStats(windowSize: 300)
    var bimanual = RollingStats(windowSize: 300)

    // Counters
    var totalFrames: Int = 0
    var framesWithHands: Int = 0
    var framesBimanual: Int = 0

    // Timing
    private var lastFrameTime: CFTimeInterval?
    private let startTime: CFTimeInterval

    // Graphs
    private let latencyGraph = Timegraph(width: 160, height: 40, lineColor: StatsColors.accentCyan)
    private let fpsGraph = Timegraph(width: 160, height: 40, lineColor: StatsColors.accentGreen)
    private let confidenceGraph = Timegraph(width: 160, height: 40, lineColor: StatsColors.accentOrange)

    // Overlay dimensions
    let overlayWidth = 280
    let overlayHeight = 220

    init() {
        startTime = CACurrentMediaTime()
    }

    func startFrame() -> CFTimeInterval {
        CACurrentMediaTime()
    }

    func endFrame(startTime: CFTimeInterval, hands: [HandLandmarks], body: BodyLandmarks?) {
        let endTime = CACurrentMediaTime()

        // Latency
        let latencyMs = (endTime - startTime) * 1000
        latency.add(latencyMs)

        // FPS
        if let lastTime = lastFrameTime {
            let dt = endTime - lastTime
            if dt > 0 {
                fpsHistory.add(1.0 / dt)
            }
        }
        lastFrameTime = endTime

        // Detection stats
        totalFrames += 1
        updateDetectionStats(hands: hands, body: body)
    }

    private func updateDetectionStats(hands: [HandLandmarks], body: BodyLandmarks?) {
        var leftPresent = false
        var rightPresent = false
        var maxHandConf: Double = 0

        for hand in hands {
            if hand.chirality == "left" {
                leftPresent = true
            } else {
                rightPresent = true
            }

            // Average confidence across joints
            var confidences: [Double] = []
            let allJoints: [Point2D?] = [
                hand.wrist,
                hand.thumbCMC, hand.thumbMP, hand.thumbIP, hand.thumbTip,
                hand.indexMCP, hand.indexPIP, hand.indexDIP, hand.indexTip,
                hand.middleMCP, hand.middlePIP, hand.middleDIP, hand.middleTip,
                hand.ringMCP, hand.ringPIP, hand.ringDIP, hand.ringTip,
                hand.littleMCP, hand.littlePIP, hand.littleDIP, hand.littleTip,
            ]
            for joint in allJoints {
                if let j = joint {
                    confidences.append(Double(j.confidence))
                }
            }
            if !confidences.isEmpty {
                let avgConf = confidences.reduce(0, +) / Double(confidences.count)
                maxHandConf = max(maxHandConf, avgConf)
            }
        }

        leftHandPresent.add(leftPresent ? 1.0 : 0.0)
        rightHandPresent.add(rightPresent ? 1.0 : 0.0)

        let isBimanual = leftPresent && rightPresent
        bimanual.add(isBimanual ? 1.0 : 0.0)

        if leftPresent || rightPresent {
            framesWithHands += 1
            handConfidence.add(maxHandConf)
        } else {
            // No hands detected - add zero confidence so graph keeps running
            handConfidence.add(0.0)
        }

        if isBimanual {
            framesBimanual += 1
        }
    }

    var runtime: Double {
        CACurrentMediaTime() - startTime
    }

    // MARK: - Rendering

    func renderOverlay(onto pixelBuffer: CVPixelBuffer, atX originX: Int, atY originY: Int) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return }

        // Flip for Core Graphics coordinate system
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)

        let x = originX
        let y = originY

        // Background with border
        context.setFillColor(StatsColors.bg)
        context.fill(CGRect(x: x, y: y, width: overlayWidth, height: overlayHeight))

        context.setStrokeColor(StatsColors.border)
        context.setLineWidth(1)
        context.stroke(CGRect(x: x, y: y, width: overlayWidth, height: overlayHeight))

        // Title bar
        context.setFillColor(StatsColors.bgLight)
        context.fill(CGRect(x: x + 1, y: y + 1, width: overlayWidth - 2, height: 24))

        drawText(context: context, text: "Pipeline Stats", x: x + 10, y: y + 6, size: 12, color: StatsColors.text)

        // Separator
        context.setStrokeColor(StatsColors.border)
        context.move(to: CGPoint(x: x, y: y + 25))
        context.addLine(to: CGPoint(x: x + overlayWidth, y: y + 25))
        context.strokePath()

        var rowY = y + 30

        // === Latency ===
        rowY = drawStatRow(
            context: context,
            x: x, y: rowY,
            label: "LATENCY",
            labelColor: StatsColors.accentCyan,
            graph: latencyGraph,
            values: latency.allValues,
            minVal: 0, maxVal: 80,
            value1: latency.p50, value1Label: "P50",
            value2: latency.p99, value2Label: "P99",
            unit: "ms"
        )

        // === FPS ===
        rowY = drawStatRow(
            context: context,
            x: x, y: rowY,
            label: "FPS",
            labelColor: StatsColors.accentGreen,
            graph: fpsGraph,
            values: fpsHistory.allValues,
            minVal: 0, maxVal: 60,
            value1: fpsHistory.latest, value1Label: "Now",
            value2: fpsHistory.p50, value2Label: "Avg",
            unit: ""
        )

        // === Confidence ===
        rowY = drawStatRow(
            context: context,
            x: x, y: rowY,
            label: "CONFIDENCE",
            labelColor: StatsColors.accentOrange,
            graph: confidenceGraph,
            values: handConfidence.allValues,
            minVal: 0, maxVal: 1.0,
            value1: handConfidence.p50, value1Label: "P50",
            value2: handConfidence.p1, value2Label: "P1",
            unit: ""
        )

        // Separator
        context.setStrokeColor(StatsColors.border)
        context.move(to: CGPoint(x: x + 10, y: rowY))
        context.addLine(to: CGPoint(x: x + overlayWidth - 10, y: rowY))
        context.strokePath()
        rowY += 8

        // === Hand Presence ===
        drawText(context: context, text: "HAND PRESENCE", x: x + 12, y: rowY, size: 9, color: StatsColors.accentBlue)
        rowY += 14

        let leftPct = (leftHandPresent.mean ?? 0) * 100
        let rightPct = (rightHandPresent.mean ?? 0) * 100
        let bimanualPct = (bimanual.mean ?? 0) * 100

        let barWidth = 70
        let barHeight = 10

        // Left hand
        drawText(context: context, text: "L", x: x + 12, y: rowY, size: 9, color: StatsColors.textDim)
        drawProgressBar(context: context, x: x + 26, y: rowY, width: barWidth, height: barHeight, value: leftPct / 100)
        drawText(context: context, text: String(format: "%.0f%%", leftPct), x: x + 26 + barWidth + 5, y: rowY, size: 8, color: StatsColors.text)

        // Right hand
        drawText(context: context, text: "R", x: x + 140, y: rowY, size: 9, color: StatsColors.textDim)
        drawProgressBar(context: context, x: x + 154, y: rowY, width: barWidth, height: barHeight, value: rightPct / 100)
        drawText(context: context, text: String(format: "%.0f%%", rightPct), x: x + 154 + barWidth + 5, y: rowY, size: 8, color: StatsColors.text)

        rowY += 14

        // Bimanual
        drawText(context: context, text: "Both", x: x + 12, y: rowY, size: 9, color: StatsColors.textDim)
        drawProgressBar(context: context, x: x + 42, y: rowY, width: barWidth, height: barHeight, value: bimanualPct / 100)
        drawText(context: context, text: String(format: "%.0f%%", bimanualPct), x: x + 42 + barWidth + 5, y: rowY, size: 8, color: StatsColors.text)

        // Runtime
        let runtimeStr = String(format: "%.0fs", runtime)
        drawText(context: context, text: runtimeStr, x: x + overlayWidth - 30, y: y + overlayHeight - 14, size: 8, color: StatsColors.textDim)
    }

    private func drawStatRow(
        context: CGContext,
        x: Int, y: Int,
        label: String,
        labelColor: CGColor,
        graph: Timegraph,
        values: [Double],
        minVal: Double, maxVal: Double,
        value1: Double?, value1Label: String,
        value2: Double?, value2Label: String,
        unit: String
    ) -> Int {
        // Label
        drawText(context: context, text: label, x: x + 12, y: y, size: 10, color: labelColor)

        // Graph
        let graphY = y + 14
        graph.render(values: values, minVal: minVal, maxVal: maxVal, context: context, x: x + 12, y: graphY)

        // Values
        let valuesX = x + 12 + graph.width + 12

        if let v1 = value1 {
            let text1 = String(format: "%@: %.1f%@", value1Label, v1, unit)
            drawText(context: context, text: text1, x: valuesX, y: graphY + 4, size: 9, color: StatsColors.text)
        }

        if let v2 = value2 {
            let text2 = String(format: "%@: %.1f%@", value2Label, v2, unit)
            drawText(context: context, text: text2, x: valuesX, y: graphY + 18, size: 9, color: StatsColors.textDim)
        }

        return y + graph.height + 20
    }

    private func drawText(context: CGContext, text: String, x: Int, y: Int, size: CGFloat, color: CGColor) {
        let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
        let attributes: [CFString: Any] = [
            kCTFontAttributeName: font,
            kCTForegroundColorAttributeName: color
        ]
        let attrString = CFAttributedStringCreate(nil, text as CFString, attributes as CFDictionary)!
        let line = CTLineCreateWithAttributedString(attrString)

        context.saveGState()
        // Flip text right-side-up (context is already flipped for top-left origin)
        context.translateBy(x: CGFloat(x), y: CGFloat(y) + size)
        context.scaleBy(x: 1, y: -1)
        context.textPosition = CGPoint(x: 0, y: 0)
        CTLineDraw(line, context)
        context.restoreGState()
    }

    private func drawProgressBar(context: CGContext, x: Int, y: Int, width: Int, height: Int, value: Double) {
        let rect = CGRect(x: x, y: y, width: width, height: height)

        // Background
        context.setFillColor(StatsColors.bgLight)
        context.fill(rect)

        // Fill
        let fillWidth = Int(Double(width) * min(1, max(0, value)))
        if fillWidth > 0 {
            context.setFillColor(StatsColors.accentBlue)
            context.fill(CGRect(x: x, y: y, width: fillWidth, height: height))
        }
    }

    // MARK: - Stats Frame Output

    func toStatsFrame(frameNumber: Int, cameraId: String) -> StatsFrame {
        return StatsFrame(
            timestamp: runtime,
            frameNumber: frameNumber,
            sourceCamera: cameraId,
            latencyP50: latency.p50,
            latencyP99: latency.p99,
            latencyLatest: latency.latest,
            fpsP50: fpsHistory.p50,
            fpsLatest: fpsHistory.latest,
            confidenceP50: handConfidence.p50,
            confidenceP1: handConfidence.p1,
            confidenceLatest: handConfidence.latest,
            leftHandPresence: leftHandPresent.mean,
            rightHandPresence: rightHandPresent.mean,
            bimanualPresence: bimanual.mean,
            totalFrames: totalFrames,
            framesWithHands: framesWithHands,
            framesBimanual: framesBimanual,
            runtime: runtime
        )
    }

    // MARK: - Summary

    func printSummary() {
        let avgFps = totalFrames > 0 ? Double(totalFrames) / runtime : 0

        fputs("\n==================================================\n", stderr)
        fputs("PIPELINE STATISTICS SUMMARY\n", stderr)
        fputs("==================================================\n", stderr)
        fputs(String(format: "Runtime:        %.1fs\n", runtime), stderr)
        fputs(String(format: "Total Frames:   %d\n", totalFrames), stderr)
        fputs(String(format: "Average FPS:    %.1f\n", avgFps), stderr)
        fputs("\n", stderr)
        fputs("Latency:\n", stderr)
        if let p50 = latency.p50 {
            fputs(String(format: "  P50:          %.1fms\n", p50), stderr)
        }
        if let p99 = latency.p99 {
            fputs(String(format: "  P99:          %.1fms\n", p99), stderr)
        }
        fputs("\n", stderr)
        fputs("Detection Confidence:\n", stderr)
        if let p50 = handConfidence.p50 {
            fputs(String(format: "  P50:          %.3f\n", p50), stderr)
        }
        if let p1 = handConfidence.p1 {
            fputs(String(format: "  P1 (worst):   %.3f\n", p1), stderr)
        }
        fputs("\n", stderr)
        fputs("Hand Activity:\n", stderr)
        fputs(String(format: "  Left Hand:    %.1f%%\n", (leftHandPresent.mean ?? 0) * 100), stderr)
        fputs(String(format: "  Right Hand:   %.1f%%\n", (rightHandPresent.mean ?? 0) * 100), stderr)
        fputs(String(format: "  Bimanual:     %.1f%%\n", (bimanual.mean ?? 0) * 100), stderr)
        fputs("==================================================\n", stderr)
    }
}
