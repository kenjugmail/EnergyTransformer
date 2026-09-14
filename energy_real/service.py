"""Small HTTP adapter for the existing local LoopTest service, without importing it."""
from __future__ import annotations
import json
import math
import urllib.error
import urllib.parse
import urllib.request


class ServiceUnavailable(RuntimeError):
    pass


class LocalModel:
    def __init__(self, base_url='http://127.0.0.1:8100', timeout=60.):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Provide an HTTP(S) service URL without embedded credentials')
        if not math.isfinite(timeout) or timeout <= 0: raise ValueError('Finite positive timeout required')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.base_url + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(4*1024*1024 + 1)
            if len(raw) > 4*1024*1024: raise ValueError('Service response exceeds limit')
            result = json.loads(raw)
            if not isinstance(result, dict): raise ValueError('Service response must be an object')
            return result
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServiceUnavailable(f'Local model service unavailable: {exc}') from exc

    def health(self):
        result = self._request('/health')
        if not isinstance(result.get('model'), str): raise ValueError('Health response has no model identity')
        return result

    def chat(self, messages, max_tokens=256):
        response = self._request('/v1/chat/completions', {
            'messages': messages, 'max_tokens': max_tokens, 'temperature': 0., 'top_p': 1.})
        try:
            text = response['choices'][0]['message']['content']
            if not isinstance(text, str): raise TypeError()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError('Malformed chat response') from exc
        return {'text': text, 'model': response.get('model'), 'usage': response.get('usage')}

    def score(self, question, options):
        if not options or not all(isinstance(x, str) for x in options): raise ValueError('Nonempty text options required')
        result = self._request('/v1/score', {'question': question, 'options': options})
        if 'answer' not in result: raise ValueError('Malformed scoring response')
        return result

    def structured(self, instruction, text):
        result = self.chat([{'role':'system','content':instruction + ' Return only a JSON object.'},
                            {'role':'user','content':text}])
        try:
            obj = json.loads(result['text'])
        except json.JSONDecodeError as exc:
            raise ValueError('Model returned invalid structured JSON; no changes applied') from exc
        if not isinstance(obj,dict): raise ValueError('Expected a structured object')
        return obj
