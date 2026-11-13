# Scopo: mantenere offline i test CPU di tutti i componenti, inclusa la CNN spaziale V1.
# Fasi: blocco HTTP esterno/Python socket/Rasterio remoto; restano ammessi file e server loopback.
# Input: fixture pytest; output: errori espliciti prima di una connessione esterna, mai skip.
# Parametri: applicazione automatica; i mock di acquisizione sintetica restano utilizzabili.
# Risorse: nessun dato, token, checkpoint o GPU; non è un sandbox per processi nativi arbitrari.
# Ripresa: monkeypatch ripristina ogni funzione a fine test; nessun file produttivo è modificato.
import ipaddress
import socket
from urllib.parse import urlsplit

import pytest
import rasterio
import requests


def loopback(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_local(host):
    if not loopback(host):
        raise RuntimeError(f'Test CPU offline: connessione esterna vietata ({host})')


@pytest.fixture(autouse=True)
def offline_cpu_tests(monkeypatch):
    original_request = requests.sessions.Session.request
    original_connection = socket.create_connection
    original_connect = socket.socket.connect
    original_open = rasterio.open

    def request(session, method, url, *args, **kwargs):
        require_local(urlsplit(url).hostname)
        return original_request(session, method, url, *args, **kwargs)

    def connection(address, *args, **kwargs):
        require_local(address[0])
        return original_connection(address, *args, **kwargs)

    def connect(sock, address):
        if isinstance(address, tuple):
            require_local(address[0])
        return original_connect(sock, address)

    def raster_open(path, *args, **kwargs):
        text = str(path)
        if text.startswith('/vsicurl/'):
            require_local(urlsplit(text[len('/vsicurl/'):]).hostname)
        elif text.startswith(('http://', 'https://')):
            require_local(urlsplit(text).hostname)
        elif text.startswith(('/vsis3/', '/vsiaz/', 's3://', 'az://')):
            raise RuntimeError('Test CPU offline: raster cloud vietato')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, 'request', request)
    monkeypatch.setattr(socket, 'create_connection', connection)
    monkeypatch.setattr(socket.socket, 'connect', connect)
    monkeypatch.setattr(rasterio, 'open', raster_open)
