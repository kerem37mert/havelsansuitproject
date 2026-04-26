import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import classes from "./methodology.module.scss";

const Methodology = () => {
  const [content, setContent] = useState("");

  useEffect(() => {
    fetch("/methodology.md")
      .then((res) => res.text())
      .then((text) => setContent(text));
  }, []);

  return (
    <div className={ classes.methodology }>
      <ReactMarkdown>
        {content}
      </ReactMarkdown>
    </div>
  );
}

export default Methodology;