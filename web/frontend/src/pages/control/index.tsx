import { useEffect, useRef, useState } from "react";
import classes from "./style.module.scss";

const WHEP_URL = "https://api.havelsansuitproject.dev/camera/whep";
const WS_URL   = "wss://api.havelsansuitproject.dev/ws/control";

type Result = {
  level: number;
  ema_score: number;
  perclos: number;
  eye_streak: boolean;
  smooth_yawn: number;
  yawn_drowsy: boolean;
  pitch: number;
  yaw: number;
  face_found: boolean;
};

const LEVEL_CONFIG = [
  { label: "Uyanık",    color: "#00c853" },
  { label: "Uyarı",     color: "#ff9100" },
  { label: "Uyukluyor", color: "#d50000" },
];

const Control = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [result, setResult] = useState<Result | null>(null);

  // ── WebRTC video stream ─────────────────────────────────────────────────
  useEffect(() => {
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    pc.ontrack = (e) => {
      if (videoRef.current) {
        videoRef.current.srcObject = e.streams[0];
      }
    };

    pc.addTransceiver("video", { direction: "recvonly" });

    pc.createOffer()
      .then((offer) => pc.setLocalDescription(offer))
      .then(() =>
        fetch(WHEP_URL, {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: pc.localDescription?.sdp,
        })
      )
      .then((res) => res.text())
      .then((sdp) =>
        pc.setRemoteDescription({ type: "answer", sdp })
      );

    return () => pc.close();
  }, []);

  // ── Backend tahmin akışı ────────────────────────────────────────────────
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (e) => setResult(JSON.parse(e.data));
    return () => ws.close();
  }, []);

  const level = result?.level ?? 0;
  const cfg   = LEVEL_CONFIG[level];
  const score = result?.ema_score ?? 0;

  const statusLabel = !result
    ? "Bekleniyor..."
    : !result.face_found
      ? "Yüz Bulunamadı"
      : cfg.label;

  return (
    <div className={classes.container}>
      <div className={classes.camera}>
        <video
          ref={videoRef}
          autoPlay muted playsInline
          style={{ width: 640, height: 360, objectFit: "contain", background: "#000" }}
        />
      </div>

      <div className={classes.panel}>
        <div className={classes.statusBar} style={{ backgroundColor: cfg.color }}>
          {statusLabel}
        </div>

        <div className={classes.scoreTrack}>
          <div className={classes.scoreFill} style={{ width: `${score}%`, backgroundColor: cfg.color }} />
          <span className={classes.scoreLabel}>{score} / 100</span>
        </div>

        {result && (
          <div className={classes.metrics}>
            <Metric label="PERCLOS"   value={result.perclos}     unit="%" max={100} warn={25} alert={40} />
            <Metric label="Esneme"    value={result.smooth_yawn} unit="%" max={100} warn={55} />
            <Metric label="Baş Pitch" value={Math.abs(result.pitch)} unit="°" max={40} warn={15} alert={25} />

            <div className={classes.textMetrics}>
              <span>Pitch: <b>{result.pitch > 0 ? "+" : ""}{result.pitch}°</b></span>
              <span>Yaw: <b>{result.yaw > 0 ? "+" : ""}{result.yaw}°</b></span>
              <span style={{ color: result.eye_streak ? "#d50000" : "#00c853" }}>
                Microsleep: <b>{result.eye_streak ? "EVET" : "Yok"}</b>
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

type MetricProps = {
  label: string;
  value: number;
  max: number;
  unit: string;
  warn?: number;
  alert?: number;
};

const Metric = ({ label, value, max, unit, warn, alert }: MetricProps) => {
  const pct   = Math.min(value / max, 1) * 100;
  const color =
    alert !== undefined && value >= alert ? "#d50000" :
    warn  !== undefined && value >= warn  ? "#ff9100" : "#00c853";

  return (
    <div className={classes.metric}>
      <div className={classes.metricHeader}>
        <span>{label}</span>
        <span style={{ color }}>{value.toFixed(1)}{unit}</span>
      </div>
      <div className={classes.barTrack}>
        <div className={classes.barFill} style={{ width: `${pct}%`, backgroundColor: color }} />
        {warn  !== undefined && <div className={classes.barMark} style={{ left: `${warn  / max * 100}%` }} />}
        {alert !== undefined && <div className={classes.barMark} style={{ left: `${alert / max * 100}%` }} />}
      </div>
    </div>
  );
};

export default Control;
