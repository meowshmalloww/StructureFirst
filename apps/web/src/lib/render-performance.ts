export type RenderScaleSample = {
  currentScale: number;
  framesPerSecond: number;
  moving: boolean;
  integratedGpu: boolean;
};

/**
 * Adjust only the WebGL drawing buffer while the camera is moving. Gaussian
 * count, covariance, opacity, sorting, and scene data are never reduced. The
 * native device-pixel ratio is restored as soon as navigation settles.
 */
export function nextRenderScale(sample: RenderScaleSample): number {
  // Keep enough native pixels that thin walls and low-opacity Gaussian tails do
  // not appear to vanish during motion. 30 FPS is an acceptable rescue-view
  // target; detail is more important than chasing a high refresh rate.
  const minimum = sample.integratedGpu ? 0.72 : 0.86;
  if (!sample.moving) return clamp(sample.currentScale + 0.4, minimum, 1);
  if (sample.framesPerSecond < 22)
    return clamp(sample.currentScale - 0.08, minimum, 1);
  if (sample.framesPerSecond < 30)
    return clamp(sample.currentScale - 0.04, minimum, 1);
  if (sample.framesPerSecond > 52)
    return clamp(sample.currentScale + 0.05, minimum, 1);
  return clamp(sample.currentScale, minimum, 1);
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, Number(value.toFixed(2))));
}
