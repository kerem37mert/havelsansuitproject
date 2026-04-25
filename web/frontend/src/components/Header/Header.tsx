import { Link } from "react-router-dom";
import type { HeaderProps } from "./Header.types";
import classes from "./Header.module.scss";

const Header = ({ title }: HeaderProps) => {
  return (
    <div className={ classes.header }>
      <div className={ classes.title }>{ title }</div>
      <div className={ classes["menu-container"] }>
        <Link className={ classes["menu-link"] } to={"/"} >Ana Sayfa</Link>
        <Link className={ classes["menu-link"]} to={"/methodology"} >Yöntem</Link>
        <Link className={ classes["menu-link"]} to={"/test"} >WebCam Test</Link>
        {/* <Link className={ classes["menu-link"]} to={"/camera"} >Kamera</Link> */}
      </div>
    </div>
  );
}

export default Header;