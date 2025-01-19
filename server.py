from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

PORT = 3000

Handler = SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    httpd.serve_forever() 