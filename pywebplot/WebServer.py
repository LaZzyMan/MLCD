from http.server import BaseHTTPRequestHandler, HTTPServer
from os import path
from urllib.parse import urlparse
from threading import Thread

PORT = 4396
HOST = 'localhost'

MIME_DIC = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.geojson': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.txt': 'text/plain',
    '.avi': 'video/x-msvideo'
}

FILE_MAP = {
    '.html': 'dist',
    '.js': 'dist/js',
    '.css': 'dist/style',
    '.json': 'dist/data',
    '.png': 'dist/image',
    '.jpg': 'dist/image',
    '.gif': 'dist/image',
    '.txt': 'dist/data',
    '.avi': 'dist/data',
    '.geojson': 'dist/data'
}


class FileRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query_path = urlparse(self.path)
        filepath = query_path.path
        if filepath.endswith('/'):
            filepath += 'index.html'
        filename, file_ext = path.splitext(filepath)
        if file_ext in MIME_DIC.keys():
            mime_type = MIME_DIC[file_ext]
            try:
                with open(path.realpath(FILE_MAP[file_ext] + filepath), 'rb') as f:
                    content = f.read()
                    self.send_response(200)
                    self.send_header('Content-type', mime_type)
                    self.end_headers()
                    self.wfile.write(content)
            except IOError:
                self.send_error(404, 'File Not Found: %s' % self.path)
        else:
            self.send_error(404, 'File Not Found: %s' % self.path)

    def log_message(self, format, *args):
        return


class WebServer(HTTPServer):
    def __init__(self, port=4396, host='localhost'):
        super().__init__((host, port), FileRequestHandler)
        self._port = port
        self._host = host
        self._views = {}

    def add_view(self, title, filename):
        self._views[title] = filename
        print('%s: http://%s:%s/%s' % (title, self._host, self._port, filename))

    @property
    def views(self):
        return self._views

    @property
    def home_url(self):
        return 'http://%s:%s' % (self._host, self._port)

    def run(self):
        print("Starting server, listen at: http://%s:%s" % (self._host, self._port))
        self.serve_forever()

    def run_bk(self):
        server_thread = Thread(target=self.run)
        server_thread.daemon = True
        server_thread.start()

    def clean(self):
        self.shutdown()
        self.server_close()


def create_server():
    port = PORT
    while True:
        try:
            return WebServer(host=HOST, port=port)
        except OSError:
            port = port + 1
            continue


WEB_SERVER = create_server()


if __name__ == '__main__':
    httpd = WebServer()
    httpd.add_view(title='Test', filename='index.html')
    httpd.run_bk()
