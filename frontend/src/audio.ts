export class Microphone {
  socket?: WebSocket;
  context?: AudioContext;
  stream?: MediaStream;
  node?: AudioWorkletNode;
  onMessage: (data: any) => void;
  constructor(onMessage: (data: any) => void) { this.onMessage = onMessage; }
  async start(config: any) {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      this.context = new AudioContext();
      await this.context.audioWorklet.addModule('/pcm-worklet.js');
      await this.context.resume();
      const socket = this.socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/a01/e3/realtime`);
      await new Promise<void>((resolve, reject) => {
        socket.onerror = () => reject(new Error('无法连接实时识别服务'));
        socket.onopen = () => socket.send(JSON.stringify(config));
        socket.onclose = () => reject(new Error('实时服务连接关闭'));
        socket.onmessage = e => {
          const data = JSON.parse(e.data);
          if (data.type === 'ready') resolve();
          else if (data.type === 'error') reject(new Error(data.error));
          this.onMessage(data);
        };
      });
      const source = this.context.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.context, 'pcm16-capture');
      const mute = this.context.createGain(); mute.gain.value = 0;
      source.connect(this.node).connect(mute).connect(this.context.destination);
      this.node.port.onmessage = e => {
        if (socket.readyState === WebSocket.OPEN) {
          if (socket.bufferedAmount > 160000) { this.onMessage({ type: 'error', error: '音频发送积压超过5秒，请停止其它实验后重试' }); this.dispose(); return; }
          socket.send(e.data);
        }
      };
      socket.onclose = () => { this.releaseAudio(); this.onMessage({ type: 'closed' }); };
      socket.onerror = () => this.onMessage({ type: 'error', error: '实时连接异常' });
    } catch (error) { this.dispose(); throw error; }
  }
  mark(label: string) { if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: 'mark', label })); }
  stop() { this.releaseAudio(); if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: 'stop' })); }
  releaseAudio() { this.node?.disconnect(); this.stream?.getTracks().forEach(t => t.stop()); if (this.context?.state !== 'closed') void this.context?.close(); }
  dispose() { this.releaseAudio(); this.socket?.close(); }
}
