# Base image
FROM nginx:1.25.3
USER root

# System deps
RUN apt-get update && apt-get install -y \
    git gcc make \
    libpcre3 libpcre3-dev \
    zlib1g zlib1g-dev \
    libssl-dev wget \
    procps bash ffmpeg net-tools \
    python3 python3-pip python3-venv \
    jq \
    && rm -rf /var/lib/apt/lists/*


# Prepare runtime directories for the custom Nginx
RUN mkdir -p /app/yolov8
RUN chmod -R 777 /app/yolov8
# Create web root
RUN mkdir -p /var/www/html/show && chmod -R 777 /var/www/html
RUN chmod -R 777 /var/log
RUN chmod -R 777 /var/tmp

# App directory
WORKDIR /app

# Create venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy configs & scripts
COPY livestream.py /app/livestream.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/*.sh

EXPOSE 80 1935
USER 1000

ENTRYPOINT ["/app/entrypoint.sh"]

