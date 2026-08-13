import { describe, expect, it } from "vitest";
import { nextRenderScale } from "./render-performance";

describe("adaptive Gaussian drawing buffer", () => {
  it("restores native resolution after camera movement stops", () => {
    expect(
      nextRenderScale({
        currentScale: 0.86,
        framesPerSecond: 24,
        moving: false,
        integratedGpu: false,
      }),
    ).toBe(1);
  });

  it("reduces only the motion drawing buffer when FPS is below target", () => {
    expect(
      nextRenderScale({
        currentScale: 1,
        framesPerSecond: 20,
        moving: true,
        integratedGpu: false,
      }),
    ).toBe(0.92);
  });

  it("keeps a higher quality floor on a discrete GPU", () => {
    expect(
      nextRenderScale({
        currentScale: 0.86,
        framesPerSecond: 10,
        moving: true,
        integratedGpu: false,
      }),
    ).toBe(0.86);
    expect(
      nextRenderScale({
        currentScale: 0.72,
        framesPerSecond: 10,
        moving: true,
        integratedGpu: true,
      }),
    ).toBe(0.72);
  });

  it("does not lower resolution while the measured FPS is healthy", () => {
    expect(
      nextRenderScale({
        currentScale: 1,
        framesPerSecond: 55,
        moving: true,
        integratedGpu: false,
      }),
    ).toBe(1);
  });
});
