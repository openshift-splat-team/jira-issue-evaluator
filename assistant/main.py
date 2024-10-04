import json
import ollama
from http.server import SimpleHTTPRequestHandler, HTTPServer

class QueryHandler(SimpleHTTPRequestHandler):
    def do_POST(self):        
        length = int(self.headers['Content-Length'])
        messagecontent = self.rfile.read(length)
        response = getStoryRecommendation(str(messagecontent, "utf-8"))                
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')        
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))

def run(server_class=HTTPServer, handler_class=QueryHandler, port=8001):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    httpd.serve_forever()

def getStoryRecommendation(request):
    request_json = json.loads(request)

    principal = "OpenShift Engineer"
    goal = ""
    outcome = ""

    if "principal" in request_json:
        principal = request_json["principal"]

    if "goal" not in request_json:
        return "must provide the goal of the story"

    goal = request_json["goal"]

    if "outcome" not in request_json:
        return "must provide the outcome of the story"

    outcome = request_json["outcome"]

    response = ollama.chat(model='mistral:instruct', messages=[
    {
        'role': 'user',
        'content': f'You are an engineer who writes user stories. You take a principal, a goal, and a desired outcome to create a user story with a story, a description, and acceptance criteria. The principal is {principal}, the goal is {goal}, and the desired outcome is {outcome}.',
    },
    ])
    return response['message']['content']

run()