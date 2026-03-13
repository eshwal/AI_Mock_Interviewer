// processor.js
// Buffers raw Float32 mic samples into 4096-sample PCM16 chunks
// before sending to the server. Prevents tiny 128-sample packets
// that confuse Gemini's VAD and cause 1007 errors.

class PcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Int16Array(4096);
    this._bufferIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const float32Data = input[0];

    for (let i = 0; i < float32Data.length; i++) {
      // Clamp to [-1, 1] then convert to PCM16
      const clamped = Math.max(-1, Math.min(1, float32Data[i]));
      this._buffer[this._bufferIndex++] = clamped * 0x7fff;

      if (this._bufferIndex >= 4096) {
        // Transfer ownership of the buffer for zero-copy send
        const chunk = new Int16Array(this._buffer);
        this.port.postMessage(chunk.buffer, [chunk.buffer]);
        // Allocate fresh buffer for next chunk
        this._buffer = new Int16Array(4096);
        this._bufferIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-processor", PcmProcessor);