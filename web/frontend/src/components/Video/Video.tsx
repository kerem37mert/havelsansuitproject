import classes from "./Video.module.scss";
import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { Circle, Layer, Stage } from "react-konva";

type Landmark = { x: number; y: number; z: number };

type FrameResult = {
  level: number;
  ema_score: number;
  perclos: number;
  eye_streak: boolean;
  smooth_yawn: number;
  yawn_drowsy: boolean;
  pitch: number;
  yaw: number;
  face_found: boolean;
  landmarks: Landmark[];
};

type WsStatus = "connecting" | "connected" | "disconnected";

const LEVEL_CONFIG = [
  { label: "Uyanık", color: "#00c853" },
  { label: "Uyarı", color: "#ff9100" },
  { label: "Uyukluyor", color: "#d50000" },
];

const Video = () => {
  const webcamRef = useRef<Webcam | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [isShowLandmarks, setIsShowLandmarks] = useState(false);
  const [result, setResult] = useState<FrameResult | null>(null);
  const [wsStatus, setWsStatus] = useState<WsStatus>("connecting");

  const connectWS = () => {
    const ws = new WebSocket(import.meta.env.VITE_WS_URL);
    wsRef.current = ws;

    setWsStatus("connecting");

    const sendNextFrame = () => {
      if (ws.readyState !== WebSocket.OPEN) return;

      const imageSrc = webcamRef.current?.getScreenshot();
      if (imageSrc) {
        ws.send(JSON.stringify({ type: "frame", data: imageSrc }));
      } else {
        // webcam henüz hazır değil; chain'i kırma, kısa süre sonra tekrar dene
        setTimeout(sendNextFrame, 100);
      }
    };

    ws.onopen = () => {
      setWsStatus("connected");
      sendNextFrame(); // ilk frame'i gönder, sonrası onmessage zincirinden gelir

      ws.onclose = () => {
        setWsStatus("disconnected");
      };
    };

    // Response-driven: backend cevabı geldiğinde bir sonraki frame'i yolla.
    // Böylece kuyruk birikmez, sistem backend hızında çalışır.
    ws.onmessage = (event) => {
      setResult(JSON.parse(event.data));
      sendNextFrame();
    };

    ws.onerror = () => {
      setWsStatus("disconnected");
    };
  };

  useEffect(() => {
    connectWS();

    return () => {
      wsRef.current?.close();
    };
  }, []);

  const level = result?.level ?? 0;
  const cfg = LEVEL_CONFIG[level];
  const score = result?.ema_score ?? 0;

  const statusLabel = !result
    ? cfg.label
    : result.face_found === false
      ? "Yüz Bulunamadı"
      : cfg.label;

  return (
    <div className={classes["video-container"]}>
      {/* Durum başlığı */}
      <div
        className={classes["status-bar"]}
        style={{ backgroundColor: cfg.color }}
      >
        {statusLabel}
      </div>

      {/* Skor çubuğu */}
      <div className={classes["score-track"]}>
        <div
          className={classes["score-fill"]}
          style={{ width: `${score}%`, backgroundColor: cfg.color }}
        />
        <span className={classes["score-label"]}>{score} / 100</span>
      </div>

      {/* Webcam + overlay */}
      <div className={classes["cam-wrapper"]}>
        <Webcam
          audio={false}
          mirrored={true}
          width={480}
          height={360}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
        />

        {isShowLandmarks && result && result.landmarks.length > 0 && (
          <Stage
            width={480}
            height={360}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              pointerEvents: "none",
            }}
          >
            <Layer>
              {result.landmarks.map((lm, i) => (
                <Circle
                  key={i}
                  x={lm.x * 480}
                  y={lm.y * 360}
                  radius={1.5}
                  fill="white"
                />
              ))}
            </Layer>
          </Stage>
        )}

        {wsStatus !== "connected" && (
          <div className={classes["cam-overlay"]}>
            {wsStatus === "connecting"
              ? "Sunucuya bağlanılıyor..."
              : (<div>
                  <div>Bağlantı Kesildi</div>
                  <button onClick={connectWS}>Yeniden Bağlan</button>
                </div>)}
          </div>
        )}
      </div>

      {/* Metrik paneli */}
      {result && wsStatus === "connected" && (
        <div className={classes["metrics"]}>
          <Metric
            label="PERCLOS"
            value={result.perclos}
            max={100}
            unit="%"
            warn={25}
            alert={40}
            highlight={result.perclos > 25}
          />
          <Metric
            label="Esneme"
            value={result.smooth_yawn}
            max={100}
            unit="%"
            warn={55}
            highlight={result.yawn_drowsy}
          />
          <Metric
            label="Baş Pitch"
            value={Math.abs(result.pitch)}
            max={40}
            unit="°"
            warn={15}
            alert={25}
            highlight={Math.abs(result.pitch) > 15}
          />

          <div className={classes["text-metrics"]}>
            <span>
              Pitch:{" "}
              <b>
                {result.pitch > 0 ? "+" : ""}
                {result.pitch}°
              </b>
            </span>
            <span>
              Yaw:{" "}
              <b>
                {result.yaw > 0 ? "+" : ""}
                {result.yaw}°
              </b>
            </span>
            <span style={{ color: result.eye_streak ? "#d50000" : "#00c853" }}>
              Microsleep: <b>{result.eye_streak ? "EVET" : "Yok"}</b>
            </span>
          </div>
        </div>
      )}

      {/* Landmark toggle */}
      <label className={classes["landmark-toggle"]}>
        <input
          type="checkbox"
          checked={isShowLandmarks}
          onChange={() => setIsShowLandmarks((p) => !p)}
        />
        Yüz noktalarını göster
      </label>
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
  highlight: boolean;
};

const Metric = ({
  label,
  value,
  max,
  unit,
  warn,
  alert,
  highlight,
}: MetricProps) => {
  const pct = Math.min(value / max, 1) * 100;
  const color = highlight
    ? alert !== undefined && value >= alert
      ? "#d50000"
      : "#ff9100"
    : "#00c853";

  return (
    <div className={classes["metric"]}>
      <div className={classes["metric-header"]}>
        <span>{label}</span>
        <span style={{ color }}>
          {value.toFixed(1)}
          {unit}
        </span>
      </div>
      <div className={classes["bar-track"]}>
        <div
          className={classes["bar-fill"]}
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
        {warn !== undefined && (
          <div
            className={classes["bar-mark"]}
            style={{ left: `${(warn / max) * 100}%` }}
          />
        )}
        {alert !== undefined && (
          <div
            className={classes["bar-mark"]}
            style={{ left: `${(alert / max) * 100}%` }}
          />
        )}
      </div>
    </div>
  );
};

export default Video;
