import XCTest
@testable import FaceRecognitionPlugin

class FaceRecognitionTests: XCTestCase {
    func testEcho() {
        // This is an example of a functional test case for a plugin.
        // Use XCTAssert and related functions to verify your tests produce the correct results.

        let implementation = FaceRecognition()
        let value = "Hello, World!"
        let result = implementation.echo(value)

        XCTAssertEqual(value, result)
    }

    // ============================
    // YawConvention (liveness-v3) — menjaga kontrak arah challenge
    // ============================

    func testYawConventionNormalizationFollowsPlatformSign() {
        // Normalisasi = yaw mentah × platformYawSign (sifat hukum, bebas nilai konstanta).
        XCTAssertEqual(YawConvention.normalize(rawYaw: 45), 45 * YawConvention.platformYawSign)
        XCTAssertEqual(YawConvention.normalize(rawYaw: -30), -30 * YawConvention.platformYawSign)
        XCTAssertEqual(YawConvention.normalize(rawYaw: 0), 0)
    }

    func testPlatformSignForVisionIsNegative() {
        // Asumsi hasil kalibrasi P0-a untuk Apple Vision: yaw mentah positif =
        // wajah berbalik ke kanan gambar = user menoleh ke KIRI anatomisnya,
        // sehingga yaw ternormalisasi (positif = kanan anatomis) perlu ×(-1).
        // JIKA halaman "Kalibrasi Yaw" example-app menunjukkan sebaliknya pada
        // perangkat uji, perbarui konstanta YawConvention DAN test ini bersamaan.
        XCTAssertEqual(YawConvention.platformYawSign, -1)
    }

    func testYawConventionOverrideFlipsOnlyWhenNegative() {
        XCTAssertEqual(YawConvention.applyOverride(15, overrideSign: 1), 15)
        XCTAssertEqual(YawConvention.applyOverride(15, overrideSign: 0), 15)
        XCTAssertEqual(YawConvention.applyOverride(15, overrideSign: -1), -15)
        XCTAssertEqual(YawConvention.applyOverride(-15, overrideSign: -1), 15)
    }

    func testLookLeftChallengeWantsNegativeYaw() {
        // LOOK_LEFT = menoleh ke kiri anatomis user = yaw ternormalisasi NEGATIF.
        // Simulasi: user menoleh ke kiri anatomis → yaw mentah +20 (kanan gambar).
        let leftTurnNormalized = YawConvention.normalize(rawYaw: 20)
        // Dengan platformYawSign = -1, hasilnya -20 → memenuhi aturan LOOK_LEFT
        // (minYaw <= -15) dan TIDAK memenuhi LOOK_RIGHT (maxYaw >= +15).
        XCTAssertLessThanOrEqual(leftTurnNormalized, -challengeYawMinForTest)
        XCTAssertLessThan(leftTurnNormalized, challengeYawMinForTest)
    }

    /// Cermin challengeYawMin (15°) dari ChallengeTrajectory.Thresholds.
    private let challengeYawMinForTest: Float = 15

    // ============================
    // ChallengeTrajectory (liveness-v3 P1) — fixture wajib lintasan challenge
    // ============================

    private func samples(_ yaw: [Float]) -> [ChallengeTrajectory.Sample] {
        yaw.map { ChallengeTrajectory.Sample(yaw: $0, timestampMs: nil) }
    }

    func testTrajectoryLookLeftCleanTurnPasses() {
        let s = samples([0, -2, -6, -12, -18, -22, -20, -14, -8, -3, -1, 0])
        let left = ChallengeTrajectory.evaluate(samples: s, challenge: "LOOK_LEFT")
        XCTAssertTrue(left.passed, "menoleh kiri bersih harus lolos LOOK_LEFT")
        XCTAssertNil(left.failureReason)

        // Arah sebaliknya harus GAGAL — justru sebagai WRONG_DIRECTION
        let right = ChallengeTrajectory.evaluate(samples: s, challenge: "LOOK_RIGHT")
        XCTAssertFalse(right.passed)
        XCTAssertEqual(right.failureReason, "CHALLENGE_WRONG_DIRECTION")
    }

    func testTrajectoryLookRightCleanTurnPasses() {
        let s = samples([0, 2, 6, 12, 18, 22, 20, 14, 8, 3, 1, 0])
        let right = ChallengeTrajectory.evaluate(samples: s, challenge: "LOOK_RIGHT")
        XCTAssertTrue(right.passed, "menoleh kanan bersih harus lolos LOOK_RIGHT")
        XCTAssertNil(right.failureReason)

        let left = ChallengeTrajectory.evaluate(samples: s, challenge: "LOOK_LEFT")
        XCTAssertFalse(left.passed)
        XCTAssertEqual(left.failureReason, "CHALLENGE_WRONG_DIRECTION")
    }

    func testTrajectoryHeadShakeFailsBothDirections() {
        // Menggeleng kiri-kanan memenuhi deviasi kedua arah — aturan lama min/max
        // meloloskan ini; R4 (dominasi satu arah) harus menolaknya.
        let s = samples([-2, -8, -16, -22, -16, -6, 2, 10, 18, 22, 14, 4])
        for challenge in ["LOOK_LEFT", "LOOK_RIGHT"] {
            let r = ChallengeTrajectory.evaluate(samples: s, challenge: challenge)
            XCTAssertFalse(r.passed, "head-shake harus gagal \(challenge)")
            XCTAssertEqual(r.failureReason, "CHALLENGE_WRONG_DIRECTION")
            XCTAssertFalse(r.rules.singleDirection)
        }
    }

    func testTrajectoryTurnWithoutReturnFails() {
        let s = samples([0, -2, -8, -16, -22, -24, -24, -24, -24, -24, -24, -24])
        let r = ChallengeTrajectory.evaluate(samples: s, challenge: "LOOK_LEFT")
        XCTAssertFalse(r.passed, "menoleh tanpa kembali lurus harus gagal")
        XCTAssertEqual(r.failureReason, "CHALLENGE_NO_RETURN")
        XCTAssertFalse(r.rules.returnFrontal)
    }

    func testTrajectorySmallMovementFails() {
        let s = samples([0, -3, -7, -10, -12, -13, -12, -10, -7, -3, -1, 0])
        let r = ChallengeTrajectory.evaluate(samples: s, challenge: "LOOK_LEFT")
        XCTAssertFalse(r.passed, "gerakan < 15° tidak membuktikan challenge")
        XCTAssertEqual(r.failureReason, "CHALLENGE_FAILED")
        XCTAssertFalse(r.rules.deviation)
    }

    func testTrajectorySingleFrameSpikeFails() {
        // Spike 1 frame memenuhi deviasi mentah, tapi bukan gerakan —
        // R5 (median-3 + run delta) harus menolaknya.
        let s = samples([0, 0, -18, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        let r = ChallengeTrajectory.evaluate(samples: s, challenge: "LOOK_LEFT")
        XCTAssertFalse(r.passed, "spike 1 frame harus gagal")
        XCTAssertEqual(r.failureReason, "CHALLENGE_FAILED")
        XCTAssertTrue(r.rules.deviation, "deviasi mentah memang terpenuhi")
        XCTAssertFalse(r.rules.smoothMotion, "tapi bukan gerakan yang halus/berkelanjutan")
    }

    func testTrajectoryPerfectVideoLoopPassesButIsFlaggedTooSmooth() {
        // Lintasan konstan-sempurna (delta tetap) = pola video-loop. Aturan
        // lintasan P1 TIDAK menolaknya (P2.3 yang menanganinya), tapi
        // deltaVariance ≈ 0 harus menjadi sinyal tooSmooth.
        let s = samples([0, -4, -8, -12, -16, -20, -24, -20, -16, -12, -8, -4])
        let r = ChallengeTrajectory.evaluate(samples: s, challenge: "LOOK_LEFT")
        XCTAssertTrue(r.passed)
        XCTAssertLessThan(r.deltaVariance, 0.01, "kecepatan konstan → deltaVariance ≈ 0")
        XCTAssertTrue(r.tooSmooth, "flag tooSmooth harus menyala untuk video-loop")

        // Gerakan manusia (mengakselerasi/melambat) → deltaVariance > 0
        let human = ChallengeTrajectory.evaluate(
            samples: samples([0, -2, -6, -12, -18, -22, -20, -14, -8, -3, -1, 0]),
            challenge: "LOOK_LEFT")
        XCTAssertGreaterThan(human.deltaVariance, 0.5)
    }

    func testTrajectoryEmptySamplesFails() {
        let r = ChallengeTrajectory.evaluate(samples: [], challenge: "LOOK_LEFT")
        XCTAssertFalse(r.passed)
        XCTAssertEqual(r.failureReason, "CHALLENGE_FAILED")
    }

    // ============================
    // ScreenSuspect (P2.2) — fixture sintetis deterministik
    // ============================

    func testScreenSuspectPeriodicityDetectsStripes() {
        // Stripes abu-abu period 4 (pola grid/moiré layar)
        var pixels = [UInt32](repeating: 0xFF202020, count: 32 * 32)
        for y in 0..<32 {
            for x in 0..<32 where (x / 4) % 2 != 0 {
                pixels[y * 32 + x] = 0xFFE0E0E0
            }
        }
        let r = ScreenSuspect.analyze(argb: pixels, width: 32, height: 32)
        XCTAssertGreaterThan(r.periodicity, 0.7, "stripes periodik harus terdeteksi")
    }

    func testScreenSuspectPeriodicityLowForNoise() {
        // Noise deterministik (LCG) — tidak boleh dianggap periodik
        var pixels = [UInt32](repeating: 0, count: 32 * 32)
        var seed: UInt32 = 42
        for i in 0..<pixels.count {
            seed = seed &* 1103515245 &+ 12345
            let red = (seed >> 16) & 0xFF
            let green = (seed >> 8) & 0xFF
            let blue = seed & 0xFF
            pixels[i] = 0xFF000000 | (red << 16) | (green << 8) | blue
        }
        let r = ScreenSuspect.analyze(argb: pixels, width: 32, height: 32)
        XCTAssertLessThan(r.periodicity, 0.5, "noise harus periodisitasnya rendah")
        XCTAssertLessThan(r.flatFraction, 0.3, "noise tidak boleh dianggap panel datar")
    }

    func testScreenSuspectFlatFractionDetectsUniformPanel() {
        let pixels = [UInt32](repeating: 0xFF808080, count: 32 * 32)
        let r = ScreenSuspect.analyze(argb: pixels, width: 32, height: 32)
        XCTAssertEqual(r.flatFraction, 1, accuracy: 0.001)
        XCTAssertEqual(r.saturationFlat, 1, accuracy: 0.001)
        XCTAssertEqual(r.score, 1, accuracy: 0.001)
    }

    func testScreenSuspectSaturationFlatLowForSaturatedColors() {
        var pixels = [UInt32](repeating: 0xFFFF0000, count: 32 * 32)
        for i in 0..<pixels.count where i % 2 != 0 {
            pixels[i] = 0xFF0000FF
        }
        let r = ScreenSuspect.analyze(argb: pixels, width: 32, height: 32)
        XCTAssertLessThan(r.saturationFlat, 0.05, "warna jenuh bukan sinyal layar-abu")
    }

    func testScreenSuspectInvalidInputIsZero() {
        let r = ScreenSuspect.analyze(argb: [], width: 0, height: 0)
        XCTAssertEqual(r.score, 0, accuracy: 0.001)
    }
}
