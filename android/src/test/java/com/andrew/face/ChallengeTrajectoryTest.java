package com.andrew.face;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

/**
 * ChallengeTrajectory (liveness-v3 P1) — fixture wajib lintasan challenge.
 * Paritas dari FaceRecognitionTests (iOS).
 */
public class ChallengeTrajectoryTest {

    private static List<ChallengeTrajectory.Sample> samples(float... yaw) {
        List<ChallengeTrajectory.Sample> list = new ArrayList<>();
        for (float y : yaw) {
            list.add(new ChallengeTrajectory.Sample(y, -1));
        }
        return list;
    }

    @Test
    public void lookLeftCleanTurnPasses() {
        List<ChallengeTrajectory.Sample> s =
                samples(0, -2, -6, -12, -18, -22, -20, -14, -8, -3, -1, 0);
        ChallengeTrajectory.Result left = ChallengeTrajectory.evaluate(s, "LOOK_LEFT");
        assertTrue("menoleh kiri bersih harus lolos LOOK_LEFT", left.passed);
        assertNull(left.failureReason);

        // Arah sebaliknya harus GAGAL — justru sebagai WRONG_DIRECTION
        ChallengeTrajectory.Result right = ChallengeTrajectory.evaluate(s, "LOOK_RIGHT");
        assertFalse(right.passed);
        assertEquals("CHALLENGE_WRONG_DIRECTION", right.failureReason);
    }

    @Test
    public void lookRightCleanTurnPasses() {
        List<ChallengeTrajectory.Sample> s =
                samples(0, 2, 6, 12, 18, 22, 20, 14, 8, 3, 1, 0);
        ChallengeTrajectory.Result right = ChallengeTrajectory.evaluate(s, "LOOK_RIGHT");
        assertTrue("menoleh kanan bersih harus lolos LOOK_RIGHT", right.passed);
        assertNull(right.failureReason);

        ChallengeTrajectory.Result left = ChallengeTrajectory.evaluate(s, "LOOK_LEFT");
        assertFalse(left.passed);
        assertEquals("CHALLENGE_WRONG_DIRECTION", left.failureReason);
    }

    @Test
    public void headShakeFailsBothDirections() {
        // Menggeleng kiri-kanan memenuhi deviasi kedua arah — aturan lama min/max
        // meloloskan ini; R4 (dominasi satu arah) harus menolaknya.
        List<ChallengeTrajectory.Sample> s =
                samples(-2, -8, -16, -22, -16, -6, 2, 10, 18, 22, 14, 4);
        for (String challenge : new String[] {"LOOK_LEFT", "LOOK_RIGHT"}) {
            ChallengeTrajectory.Result r = ChallengeTrajectory.evaluate(s, challenge);
            assertFalse("head-shake harus gagal " + challenge, r.passed);
            assertEquals("CHALLENGE_WRONG_DIRECTION", r.failureReason);
            assertFalse(r.singleDirection);
        }
    }

    @Test
    public void turnWithoutReturnFails() {
        List<ChallengeTrajectory.Sample> s =
                samples(0, -2, -8, -16, -22, -24, -24, -24, -24, -24, -24, -24);
        ChallengeTrajectory.Result r = ChallengeTrajectory.evaluate(s, "LOOK_LEFT");
        assertFalse("menoleh tanpa kembali lurus harus gagal", r.passed);
        assertEquals("CHALLENGE_NO_RETURN", r.failureReason);
        assertFalse(r.returnFrontal);
    }

    @Test
    public void smallMovementFails() {
        List<ChallengeTrajectory.Sample> s =
                samples(0, -3, -7, -10, -12, -13, -12, -10, -7, -3, -1, 0);
        ChallengeTrajectory.Result r = ChallengeTrajectory.evaluate(s, "LOOK_LEFT");
        assertFalse("gerakan < 15° tidak membuktikan challenge", r.passed);
        assertEquals("CHALLENGE_FAILED", r.failureReason);
        assertFalse(r.deviation);
    }

    @Test
    public void singleFrameSpikeFails() {
        // Spike 1 frame memenuhi deviasi mentah, tapi bukan gerakan —
        // R5 (median-3 + run delta) harus menolaknya.
        List<ChallengeTrajectory.Sample> s =
                samples(0, 0, -18, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        ChallengeTrajectory.Result r = ChallengeTrajectory.evaluate(s, "LOOK_LEFT");
        assertFalse("spike 1 frame harus gagal", r.passed);
        assertEquals("CHALLENGE_FAILED", r.failureReason);
        assertTrue("deviasi mentah memang terpenuhi", r.deviation);
        assertFalse("tapi bukan gerakan yang halus/berkelanjutan", r.smoothMotion);
    }

    @Test
    public void perfectVideoLoopPassesButIsFlaggedTooSmooth() {
        // Lintasan konstan-sempurna (delta tetap) = pola video-loop. Aturan
        // lintasan P1 TIDAK menolaknya (P2.3 yang menanganinya), tapi
        // deltaVariance ≈ 0 harus menjadi sinyal tooSmooth.
        List<ChallengeTrajectory.Sample> s =
                samples(0, -4, -8, -12, -16, -20, -24, -20, -16, -12, -8, -4);
        ChallengeTrajectory.Result r = ChallengeTrajectory.evaluate(s, "LOOK_LEFT");
        assertTrue(r.passed);
        assertTrue("kecepatan konstan → deltaVariance ≈ 0", r.deltaVariance < 0.01f);
        assertTrue("flag tooSmooth harus menyala untuk video-loop", r.tooSmooth);

        // Gerakan manusia (mengakselerasi/melambat) → deltaVariance > 0
        ChallengeTrajectory.Result human = ChallengeTrajectory.evaluate(
                samples(0, -2, -6, -12, -18, -22, -20, -14, -8, -3, -1, 0), "LOOK_LEFT");
        assertTrue(human.deltaVariance > 0.5f);
    }

    @Test
    public void emptySamplesFail() {
        ChallengeTrajectory.Result r =
                ChallengeTrajectory.evaluate(new ArrayList<>(), "LOOK_LEFT");
        assertFalse(r.passed);
        assertEquals("CHALLENGE_FAILED", r.failureReason);
    }
}
