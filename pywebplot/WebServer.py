from http.server import BaseHTTPRequestHandler, HTTPServer
from os import path
from urllib.parse import urlparse

MIME_DIC = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
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
    '.geojson': 'dist/data',
    '.npy': 'dist/data'
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


class WebServer(HTTPServer):
    def __init__(self, port=4396, host='localhost'):
        super().__init__((host, port), FileRequestHandler)
        self._port = port
        self._host = host
        self._views = {}

    def add_view(self, title, filename):
        self._views[title] = filename

    @property
    def views(self):
        return self._views

    @property
    def home_url(self):
        return 'http://%s:%s' % (self._host, self._port)

    def run(self):
        print("Starting server, listen at: http://%s:%s" % (self._host, self._port))
        for key, item in self.views.items():
            print('%s: http://%s:%s/%s' % (key, self._host, self._port, item))
        self.serve_forever()


if __name__ == '__main__':
    httpd = WebServer()
    httpd.add_view(title='Test', filename='index.html')
    httpd.run()
