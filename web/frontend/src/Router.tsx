import { BrowserRouter, Route, Routes } from "react-router-dom";
import Header from "./components/Header";
import Webcam from "./pages/Webcam";
import Methodology from "./pages/Methodology";
import Control from "./pages/control";
import Camera from "./pages/camera";

const Router = () => {
  return (
    <BrowserRouter>
      <Header title="Havelsan Suit Project Demo" />
      <Routes>
        <Route index element={ <Control /> } />
        <Route path="/methodology" element={ <Methodology /> } />
        <Route path="/test" element={ <Webcam /> } />
        {/* <Route path="/camera" element={ <Camera /> } /> */}
      </Routes>
    </BrowserRouter>
  );
}

export default Router;