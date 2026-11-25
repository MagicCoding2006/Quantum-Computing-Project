FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY asset_lab.py trade_lab.py ibm_quantum.py benchmark_asset_lab.py checklist_scanner.py ./
COPY assetlab/ ./assetlab/
ENTRYPOINT ["python", "asset_lab.py"]
CMD ["--demo", "--out", "/output"]
