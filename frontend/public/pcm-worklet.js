/* Stateful area resampler. Carries fractional boundaries across render quanta. */
class PCM16Capture extends AudioWorkletProcessor {
  constructor() { super(); this.ratio = sampleRate / 16000; this.remaining = this.ratio; this.sum = 0; this.out = new Int16Array(8000); this.pos = 0; }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (const value of input) {
      let left = 1;
      while (left > 1e-9) {
        const take = Math.min(left, this.remaining);
        this.sum += value * take; left -= take; this.remaining -= take;
        if (this.remaining < 1e-9) {
          this.out[this.pos++] = Math.round(Math.max(-1, Math.min(1, this.sum / this.ratio)) * 32767);
          this.sum = 0; this.remaining = this.ratio;
          if (this.pos === this.out.length) { this.port.postMessage(this.out.buffer, [this.out.buffer]); this.out = new Int16Array(8000); this.pos = 0; }
        }
      }
    }
    return true;
  }
}
registerProcessor('pcm16-capture', PCM16Capture);
