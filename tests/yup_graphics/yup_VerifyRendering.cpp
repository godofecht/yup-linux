/*
  ==============================================================================

   This file is part of the YUP library.
   Copyright (c) 2026 - kunitoki@gmail.com

   YUP is an open source library subject to open-source licensing.

   The code included in this file is provided under the terms of the ISC license
   http://www.isc.org/downloads/software-support-policy/isc-license. Permission
   To use, copy, modify, and/or distribute this software for any purpose with or
   without fee is hereby granted provided that the above copyright notice and
   this permission notice appear in all copies.

   YUP IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, AND ALL WARRANTIES, WHETHER
   EXPRESSED OR IMPLIED, INCLUDING MERCHANTABILITY AND FITNESS FOR PURPOSE, ARE
   DISCLAIMED.

  ==============================================================================
*/

#include <gtest/gtest.h>

#include <yup_graphics/yup_graphics.h>
#include <yup_rhi/yup_rhi.h>

using namespace yup;

#if YUP_LINUX

#include <EGL/egl.h>
#include <dlfcn.h>

namespace
{

//==============================================================================
/** RAII wrapper around an EGL display, config, surface, and context.

    Creates a headless OpenGL ES 2.0+ context backed by a 1x1 pbuffer.
    On Mesa with llvmpipe (CI), this gives a real software GL context
    without requiring a window system.
*/
class EglContext
{
public:
    EglContext()
    {
        display = eglGetDisplay (EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY)
            return;

        EGLint major = 0, minor = 0;
        if (! eglInitialize (display, &major, &minor))
            return;

        if (! eglBindAPI (EGL_OPENGL_API))
            return;

        const EGLint configAttribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_NONE
        };

        EGLint numConfigs = 0;
        if (! eglChooseConfig (display, configAttribs, &config, 1, &numConfigs) || numConfigs == 0)
            return;

        const EGLint pbufferAttribs[] = {
            EGL_WIDTH, 1,
            EGL_HEIGHT, 1,
            EGL_NONE
        };

        surface = eglCreatePbufferSurface (display, config, pbufferAttribs);
        if (surface == EGL_NO_SURFACE)
            return;

        const EGLint contextAttribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL_CONTEXT_MINOR_VERSION, 3,
            EGL_NONE
        };

        context = eglCreateContext (display, config, EGL_NO_CONTEXT, contextAttribs);
        if (context == EGL_NO_CONTEXT)
            return;

        if (! eglMakeCurrent (display, surface, surface, context))
            return;

        valid = true;
    }

    ~EglContext()
    {
        if (display != EGL_NO_DISPLAY)
        {
            eglMakeCurrent (display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (context != EGL_NO_CONTEXT)
                eglDestroyContext (display, context);
            if (surface != EGL_NO_SURFACE)
                eglDestroySurface (display, surface);
            eglTerminate (display);
        }
    }

    bool isValid() const noexcept { return valid; }

    static void* loader (const char* name)
    {
        return (void*) eglGetProcAddress (name);
    }

private:
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLConfig config = {};
    EGLSurface surface = EGL_NO_SURFACE;
    EGLContext context = EGL_NO_CONTEXT;
    bool valid = false;
};

//==============================================================================
/** Fixture that creates a real GPU device via EGL on Linux.

    Skips all tests if no EGL/GL context is available (e.g. no GPU, no
    Mesa, no display). On CI with xvfb + llvmpipe, the context is real
    and all tests run.
*/
class VerifyRenderingTests : public ::testing::Test
{
protected:
    static EglContext* egl;

    static void SetUpTestSuite()
    {
        egl = new EglContext();
    }

    static void TearDownTestSuite()
    {
        delete egl;
        egl = nullptr;
    }

    void SetUp() override
    {
        if (egl == nullptr || ! egl->isValid())
            GTEST_SKIP() << "No EGL/GL context available";

        GpuDevice::Options options;
        options.loaderFunction = &EglContext::loader;

        device = GpuDevice::create (GpuPlatform::OpenGL, options);
        if (device == nullptr)
            GTEST_SKIP() << "Failed to create OpenGL GpuDevice";
    }

    GpuDevice::Ptr device;
};

EglContext* VerifyRenderingTests::egl = nullptr;

//==============================================================================
// Helper to read back all pixels from an offscreen target.
//==============================================================================
struct PixelBuffer
{
    int width;
    int height;
    std::vector<uint8_t> data;

    PixelBuffer (int w, int h)
        : width (w)
        , height (h)
        , data (static_cast<size_t> (w) * static_cast<size_t> (h) * 4, 0)
    {
    }

    // Returns RGBA at (x, y). Origin is top-left (readOffscreenPixels flips).
    uint32_t getPixel (int x, int y) const
    {
        const auto idx = (static_cast<size_t> (y) * width + x) * 4;
        return (uint32_t) data[idx] | (uint32_t) data[idx + 1] << 8
             | (uint32_t) data[idx + 2] << 16 | (uint32_t) data[idx + 3] << 24;
    }

    // Returns {R, G, B, A} at (x, y) as individual bytes.
    std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> getRGBA (int x, int y) const
    {
        const auto idx = (static_cast<size_t> (y) * width + x) * 4;
        return { data[idx], data[idx + 1], data[idx + 2], data[idx + 3] };
    }
};

//==============================================================================
// Color tolerance for software renderer (llvmpipe) pixel comparisons.
// Software rendering is exact, but we allow 1 bit of slack for rounding.
//==============================================================================
static constexpr int kTolerance = 2;

static ::testing::AssertionResult pixelMatches (const char* exprPixel, const char* exprExpected,
                                                 uint32_t pixel, uint32_t expected)
{
    const uint8_t pr = pixel & 0xFF, pg = (pixel >> 8) & 0xFF, pb = (pixel >> 16) & 0xFF, pa = (pixel >> 24) & 0xFF;
    const uint8_t er = expected & 0xFF, eg = (expected >> 8) & 0xFF, eb = (expected >> 16) & 0xFF, ea = (expected >> 24) & 0xFF;

    if (std::abs (pr - er) <= kTolerance && std::abs (pg - eg) <= kTolerance
        && std::abs (pb - eb) <= kTolerance && std::abs (pa - ea) <= kTolerance)
        return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
        << "pixel mismatch: got RGBA(" << (int) pr << "," << (int) pg << "," << (int) pb << "," << (int) pa
        << ") expected RGBA(" << (int) er << "," << (int) eg << "," << (int) eb << "," << (int) ea << ")";
}

#define EXPECT_PIXEL_EQ(pixel, expected) EXPECT_PRED_FORMAT2(pixelMatches, pixel, expected)

} // namespace

//==============================================================================
// Clear color tests: clear the offscreen target to a known color, read back
// every pixel, and verify it matches. This exercises the full pipeline:
// EGL context -> GpuDevice -> offscreen FBO -> glClear -> glReadPixels.
//==============================================================================

TEST_F (VerifyRenderingTests, ClearToRed)
{
    constexpr int w = 64, h = 64;
    auto target = device->createOffscreenTarget (w, h);
    ASSERT_NE (target, nullptr);

    EXPECT_TRUE (device->clearOffscreen (*target, GpuColor (1.0f, 0.0f, 0.0f, 1.0f)));

    PixelBuffer pb (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*target, pb.data.data(), pb.data.size()));

    // RGBA: (255, 0, 0, 255)
    const uint32_t expected = 0xFF0000FF;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            EXPECT_PIXEL_EQ (pb.getPixel (x, y), expected);
}

TEST_F (VerifyRenderingTests, ClearToGreen)
{
    constexpr int w = 64, h = 64;
    auto target = device->createOffscreenTarget (w, h);
    ASSERT_NE (target, nullptr);

    EXPECT_TRUE (device->clearOffscreen (*target, GpuColor (0.0f, 1.0f, 0.0f, 1.0f)));

    PixelBuffer pb (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*target, pb.data.data(), pb.data.size()));

    // RGBA: (0, 255, 0, 255)
    const uint32_t expected = 0xFF00FF00;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            EXPECT_PIXEL_EQ (pb.getPixel (x, y), expected);
}

TEST_F (VerifyRenderingTests, ClearToBlue)
{
    constexpr int w = 64, h = 64;
    auto target = device->createOffscreenTarget (w, h);
    ASSERT_NE (target, nullptr);

    EXPECT_TRUE (device->clearOffscreen (*target, GpuColor (0.0f, 0.0f, 1.0f, 1.0f)));

    PixelBuffer pb (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*target, pb.data.data(), pb.data.size()));

    // RGBA: (0, 0, 255, 255)
    const uint32_t expected = 0xFFFF0000;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            EXPECT_PIXEL_EQ (pb.getPixel (x, y), expected);
}

TEST_F (VerifyRenderingTests, ClearToWhite)
{
    constexpr int w = 64, h = 64;
    auto target = device->createOffscreenTarget (w, h);
    ASSERT_NE (target, nullptr);

    EXPECT_TRUE (device->clearOffscreen (*target, GpuColor::white()));

    PixelBuffer pb (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*target, pb.data.data(), pb.data.size()));

    // RGBA: (255, 255, 255, 255)
    const uint32_t expected = 0xFFFFFFFF;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            EXPECT_PIXEL_EQ (pb.getPixel (x, y), expected);
}

TEST_F (VerifyRenderingTests, ClearToBlack)
{
    constexpr int w = 64, h = 64;
    auto target = device->createOffscreenTarget (w, h);
    ASSERT_NE (target, nullptr);

    EXPECT_TRUE (device->clearOffscreen (*target, GpuColor::black()));

    PixelBuffer pb (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*target, pb.data.data(), pb.data.size()));

    // RGBA: (0, 0, 0, 255)
    const uint32_t expected = 0xFF000000;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            EXPECT_PIXEL_EQ (pb.getPixel (x, y), expected);
}

TEST_F (VerifyRenderingTests, ClearToCustomColor)
{
    constexpr int w = 32, h = 32;
    auto target = device->createOffscreenTarget (w, h);
    ASSERT_NE (target, nullptr);

    // A non-trivial color: (123, 200, 50, 255)
    EXPECT_TRUE (device->clearOffscreen (*target, GpuColor (123.0f / 255.0f, 200.0f / 255.0f, 50.0f / 255.0f, 1.0f)));

    PixelBuffer pb (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*target, pb.data.data(), pb.data.size()));

    const uint32_t expected = 0xFF32C87B; // RGBA: (123, 200, 50, 255)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            EXPECT_PIXEL_EQ (pb.getPixel (x, y), expected);
}

//==============================================================================
// Two-pass clear: clear to one color, then clear to another. Verifies the
// second clear actually overwrites the first (not a no-op).
//==============================================================================

TEST_F (VerifyRenderingTests, SecondClearOverwritesFirst)
{
    constexpr int w = 32, h = 32;
    auto target = device->createOffscreenTarget (w, h);
    ASSERT_NE (target, nullptr);

    EXPECT_TRUE (device->clearOffscreen (*target, GpuColor (1.0f, 0.0f, 0.0f, 1.0f)));
    EXPECT_TRUE (device->clearOffscreen (*target, GpuColor (0.0f, 0.0f, 1.0f, 1.0f)));

    PixelBuffer pb (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*target, pb.data.data(), pb.data.size()));

    const uint32_t expectedBlue = 0xFFFF0000; // RGBA: (0, 0, 255, 255)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            EXPECT_PIXEL_EQ (pb.getPixel (x, y), expectedBlue);
}

//==============================================================================
// Begin/end frame with clear via FrameDescriptor. This exercises the Rive
// render path, not just raw glClear.
//==============================================================================

TEST_F (VerifyRenderingTests, FrameClearViaRiveRenderer)
{
    constexpr int w = 32, h = 32;
    // Use createRenderableTarget which reserves a dedicated render context,
    // required for beginOffscreen/endOffscreen to actually drive a frame.
    // createOffscreenTarget does not set contextSlot, so beginOffscreen is a no-op.
    auto target = device->createRenderableTarget (w, h);
    ASSERT_NE (target, nullptr);

    rive::gpu::RenderContext::FrameDescriptor frameDesc;
    frameDesc.renderTargetWidth = static_cast<uint32_t> (w);
    frameDesc.renderTargetHeight = static_cast<uint32_t> (h);
    frameDesc.loadAction = rive::gpu::LoadAction::clear;
    frameDesc.clearColor = 0xFF0000FF; // Rive ColorInt: 0xAARRGGBB = opaque blue

    device->beginOffscreen (*target, frameDesc);
    device->endOffscreen (*target);

    PixelBuffer pb (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*target, pb.data.data(), pb.data.size()));

    // Verify all pixels are the same solid color.
    const uint32_t firstPixel = pb.getPixel (0, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            EXPECT_EQ (pb.getPixel (x, y), firstPixel)
                << "Pixel at (" << x << "," << y << ") differs from (0,0)";

    // Verify it's not transparent black (the default uninitialized state).
    EXPECT_NE (firstPixel, 0u);
}

//==============================================================================
// Multiple targets: verify two offscreen targets don't interfere.
//==============================================================================

TEST_F (VerifyRenderingTests, MultipleTargetsAreIndependent)
{
    constexpr int w = 16, h = 16;

    auto targetA = device->createOffscreenTarget (w, h);
    ASSERT_NE (targetA, nullptr);
    auto targetB = device->createOffscreenTarget (w, h);
    ASSERT_NE (targetB, nullptr);

    EXPECT_TRUE (device->clearOffscreen (*targetA, GpuColor (1.0f, 0.0f, 0.0f, 1.0f)));
    EXPECT_TRUE (device->clearOffscreen (*targetB, GpuColor (0.0f, 1.0f, 0.0f, 1.0f)));

    PixelBuffer pbA (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*targetA, pbA.data.data(), pbA.data.size()));

    PixelBuffer pbB (w, h);
    ASSERT_TRUE (device->readOffscreenPixels (*targetB, pbB.data.data(), pbB.data.size()));

    const uint32_t expectedRed = 0xFF0000FF;
    const uint32_t expectedGreen = 0xFF00FF00;

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            EXPECT_PIXEL_EQ (pbA.getPixel (x, y), expectedRed);
            EXPECT_PIXEL_EQ (pbB.getPixel (x, y), expectedGreen);
        }
    }
}

#else
// Non-Linux platforms: register a dummy test so the suite is not empty.
TEST (VerifyRenderingTests, NotAvailableOnThisPlatform)
{
    GTEST_SKIP() << "Deterministic visual rendering tests are Linux-only (EGL)";
}
#endif
