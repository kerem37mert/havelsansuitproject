import { BrowserRouter, Route, Routes } from "react-router-dom";
import Header from "./components/Header";
import Home from "./pages/Home";
import Methodology from "./pages/Methodology";

const Router = () => {
  return (
    <BrowserRouter>
      <Header title="Havelsan Suit Project Demo" />
      <Routes>
        <Route index element={ <Home /> } />
        <Route path="/methodology" element={ <Methodology /> } />
      </Routes>
    </BrowserRouter>
  );
}

export default Router;