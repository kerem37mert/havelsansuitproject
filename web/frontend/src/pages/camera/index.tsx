import { useEffect, useRef } from "react";

const WHEP_URL = "https://api.havelsansuitproject.dev/camera/whep";

const Camera = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

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

  return (
    <video
      ref={videoRef}
      autoPlay
      muted
      playsInline
      style={{ width: "400px", height: "400px" }}
    />
  );
};

export default Camera;
