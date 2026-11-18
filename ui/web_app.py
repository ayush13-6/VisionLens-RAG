import http.server
import json
import logging
import os
import socketserver
import urllib.parse
from pathlib import Path
from typing import Any, Dict

import config
from agent.vision_rag_agent import VisionRAGAgent

logger = logging.getLogger("VisionLensWeb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VisionRAGRequestHandler(http.server.SimpleHTTPRequestHandler):
    agent: VisionRAGAgent = None

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict:
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            return {}
        raw = self.rfile.read(content_len)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            static_file = Path(__file__).parent / "static" / "index.html"
            if static_file.exists():
                with open(static_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        elif path == "/api/status":
            status = self.agent.get_status()
            self._send_json(status)
            return

        elif path == "/api/image":
            query = urllib.parse.parse_qs(parsed.query)
            img_rel_path = query.get("path", [""])[0]
            # Resolve safe path
            img_path = Path(img_rel_path)
            if not img_path.is_absolute():
                img_path = config.BASE_DIR / img_path

            if img_path.exists() and img_path.is_file():
                ext = img_path.suffix.lower()
                content_type = "image/png" if ext == ".png" else "image/jpeg"
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(img_bytes)))
                self.end_headers()
                self.wfile.write(img_bytes)
                return
            else:
                self.send_error(404, f"Image not found: {img_rel_path}")
                return

        self.send_error(404, "Endpoint Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        body = self._read_json_body()

        if path == "/api/inspect":
            img_path = body.get("image_path", "sample_data/sample_invoice.png")
            resolved = Path(img_path)
            if not resolved.is_absolute():
                resolved = config.BASE_DIR / resolved

            try:
                result = self.agent.inspect_image(str(resolved))
                # Add preprocessed binarized paths
                prep = self.agent.tools.preprocess_image(str(resolved), base_name=resolved.stem)
                result["binarized_image_path"] = prep["binarized_image_path"]
                result["rectified_image_path"] = prep["rectified_image_path"]
                result["annotated_image_path"] = result["layout"]["annotated_image_path"]
                self._send_json(result)
            except Exception as e:
                logger.error(f"Inspect failed: {e}", exc_info=True)
                self._send_json({"error": str(e)}, status=500)
            return

        elif path == "/api/ingest":
            img_path = body.get("image_path", "sample_data/sample_invoice.png")
            title = body.get("title", "Visual Document")
            text_content = body.get("text_content")

            resolved = Path(img_path)
            if not resolved.is_absolute():
                resolved = config.BASE_DIR / resolved

            try:
                res = self.agent.scan_and_ingest(
                    image_path=str(resolved),
                    title=title,
                    text_content=text_content
                )
                self._send_json(res)
            except Exception as e:
                logger.error(f"Ingest failed: {e}", exc_info=True)
                self._send_json({"error": str(e)}, status=500)
            return

        elif path == "/api/ask":
            query = body.get("query", "").strip()
            if not query:
                self._send_json({"error": "Empty query"}, status=400)
                return

            try:
                res = self.agent.ask(query)
                self._send_json(res)
            except Exception as e:
                logger.error(f"Ask failed: {e}", exc_info=True)
                self._send_json({"error": str(e)}, status=500)
            return

        elif path == "/api/clear":
            self.agent.vector_store.clear()
            self._send_json({"status": "cleared"})
            return

        self.send_error(404, "Endpoint Not Found")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def start_server(host: str = config.WEB_HOST, port: int = config.WEB_PORT):
    VisionRAGRequestHandler.agent = VisionRAGAgent()
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, VisionRAGRequestHandler)
    print(f"\n=======================================================")
    print(f" VisionLens-RAG Web Dashboard Running at:")
    print(f" >>> http://{host}:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
