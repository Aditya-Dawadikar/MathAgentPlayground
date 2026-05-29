FROM ollama/ollama:latest

COPY start-ollama.sh /usr/local/bin/start-ollama.sh

RUN chmod +x /usr/local/bin/start-ollama.sh

ENTRYPOINT ["/usr/local/bin/start-ollama.sh"]