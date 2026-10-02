# Offline sandbox for evaluation suites (coding and data_analysis).
# Multi-arch base (linux/amd64 and linux/arm64); build on the evaluation host and
# pin the resulting image digest in the suite config. The harness never pulls.
FROM python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && python -c "import pandas, numpy, scipy, sklearn, statsmodels, matplotlib, seaborn, tabulate"
ENV MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /workspace
