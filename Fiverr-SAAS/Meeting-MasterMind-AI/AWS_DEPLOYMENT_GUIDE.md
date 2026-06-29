# MeetingMind AI — AWS Free Tier Deployment Guide

**Last Updated:** 2026-06-29  
**Stack:** FastAPI backend (Docker) + Next.js frontend + PostgreSQL (Docker) + S3 + ECR + EC2

---

## Overview

This guide deploys MeetingMind AI using AWS Free Tier services only. The architecture:

```
Internet
   │
   ▼
EC2 t2.micro (Ubuntu 22.04)
   ├── Docker: meetingmind-backend (FastAPI, port 8000)
   ├── Docker: meetingmind-db (PostgreSQL 16, port 5432)
   └── Nginx (reverse proxy, port 80/443)

Vercel (Free)
   └── Next.js frontend → calls EC2 backend

AWS S3
   └── Audio/video file storage

AWS ECR
   └── Docker image registry
```

**Free Tier limits used:**
| Service | Free Tier | Usage |
|---------|-----------|-------|
| EC2 t2.micro | 750 hrs/month × 12 months | Backend + DB |
| S3 | 5 GB storage, 20K GET, 2K PUT/month | Meeting files |
| ECR | 500 MB/month | Docker image (~800 MB; exceeds — see note) |
| AWS Transcribe | 60 min/month × 12 months | Transcription |
| Data transfer out | 1 GB/month | API responses |
| Vercel | Unlimited for hobby | Frontend |

> **ECR Note:** Your Docker image is ~800 MB, which exceeds the 500 MB ECR free tier. Either push to ECR once and keep the same image (you only pay for storage beyond 500 MB, ~$0.10/GB/month) or build the image directly on EC2 to avoid ECR costs entirely — instructions for both options are below.

---

## Prerequisites

- AWS account (free tier active): https://aws.amazon.com/free
- GitHub account with your monorepo pushed
- Vercel account (free): https://vercel.com
- SSH client (PowerShell, PuTTY, or WSL)

---

## Phase 1: AWS Account Setup

### 1.1 Create Billing Alert (do this first)

1. AWS Console → **Billing** → **Budgets** → **Create Budget**
2. Choose **Cost Budget**
3. Set **$5/month** threshold → get email alert before any surprise charges
4. Email: vparmarce@gmail.com

### 1.2 Create IAM User for Deployments

Never use root account credentials in code.

1. AWS Console → **IAM** → **Users** → **Add users**
2. Username: `meetingmind-deploy`
3. Select **Programmatic access** (Access key)
4. Attach these policies directly:
   - `AmazonS3FullAccess`
   - `AmazonEC2ContainerRegistryPowerUser`
   - `AmazonTranscribeFullAccess`
5. Download the CSV — you will never see the secret key again
6. Save:
   - `AWS_ACCESS_KEY_ID` = Access key ID
   - `AWS_SECRET_ACCESS_KEY` = Secret access key
   - `AWS_REGION` = `ap-south-1` (Mumbai — lowest latency for India)

---

## Phase 2: S3 Bucket Setup

### 2.1 Create the Bucket

1. AWS Console → **S3** → **Create bucket**
2. Bucket name: `meetingmind-meetings-<your-name>` (must be globally unique)
3. Region: `ap-south-1`
4. **Block all public access**: ON (files are served via presigned URLs)
5. Versioning: OFF (saves storage)
6. Click **Create bucket**

### 2.2 Add CORS Policy (required for direct browser uploads later)

1. Click your bucket → **Permissions** tab → **CORS**
2. Paste:

```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST", "DELETE"],
    "AllowedOrigins": [
      "http://localhost:3000",
      "https://your-app.vercel.app"
    ],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```

3. Replace `your-app.vercel.app` with your real Vercel domain once deployed

### 2.3 Add Lifecycle Rule (prevents storage bloat)

1. Bucket → **Management** → **Create lifecycle rule**
2. Rule name: `expire-old-transcripts`
3. Apply to all objects: YES
4. Transition to Glacier after 90 days (optional, saves cost)
5. No expiration needed (meetings are user data)

---

## Phase 3: ECR Repository Setup

### 3.1 Create ECR Repository

1. AWS Console → **ECR** → **Create repository**
2. Repository name: `meetingmind-backend`
3. Visibility: **Private**
4. Region: `ap-south-1`
5. Note the full URI — it looks like:
   `123456789.dkr.ecr.ap-south-1.amazonaws.com/meetingmind-backend`
   Save this as `ECR_REGISTRY_URI`

---

## Phase 4: EC2 Instance Launch

### 4.1 Launch the Instance

1. AWS Console → **EC2** → **Launch Instance**
2. **Name:** `meetingmind-server`
3. **AMI:** Ubuntu Server 22.04 LTS (Free tier eligible)
4. **Instance type:** `t2.micro` (Free tier: 1 vCPU, 1 GB RAM)
5. **Key pair:** Create new → name it `meetingmind-key` → download `.pem` file
   - Save to: `C:\Users\Varsha Parmar\.ssh\meetingmind-key.pem`
6. **Network settings:**
   - Create new security group named `meetingmind-sg`
   - Allow SSH (port 22) from **My IP** (not 0.0.0.0/0)
   - Allow HTTP (port 80) from **Anywhere**
   - Allow HTTPS (port 443) from **Anywhere**
   - Allow Custom TCP port 8000 from **Anywhere** (temporary; remove after Nginx is set up)
7. **Storage:** 20 GB gp2 (free tier allows 30 GB)
8. Click **Launch Instance**

### 4.2 Allocate Elastic IP (keeps IP stable on restart)

1. EC2 → **Elastic IPs** → **Allocate Elastic IP**
2. Region: ap-south-1 → **Allocate**
3. Select the new IP → **Actions** → **Associate Elastic IP**
4. Associate with your `meetingmind-server` instance
5. Note your **Elastic IP** — this is your server's permanent IP

### 4.3 Fix .pem File Permissions (Windows)

Open PowerShell as Administrator:

```powershell
# Remove inherited permissions and grant only your account access
icacls "C:\Users\Varsha Parmar\.ssh\meetingmind-key.pem" /inheritance:r
icacls "C:\Users\Varsha Parmar\.ssh\meetingmind-key.pem" /grant:r "$env:USERNAME:(R)"
```

### 4.4 Test SSH Connection

```powershell
ssh -i "C:\Users\Varsha Parmar\.ssh\meetingmind-key.pem" ubuntu@<YOUR_ELASTIC_IP>
```

Type `yes` when asked about fingerprint. If you see `ubuntu@ip-x-x-x-x:~$`, you're in.

---

## Phase 5: Server Software Installation

Run all commands below while SSH'd into your EC2 instance.

### 5.1 Update System

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

### 5.2 Install Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu
newgrp docker
# Verify
docker --version
```

### 5.3 Install Docker Compose Plugin

```bash
sudo apt-get install -y docker-compose-plugin
# Verify
docker compose version
```

### 5.4 Install AWS CLI

```bash
sudo apt-get install -y awscli
# Verify
aws --version
```

### 5.5 Configure AWS CLI on EC2

```bash
aws configure
# Enter:
# AWS Access Key ID: <your AWS_ACCESS_KEY_ID>
# AWS Secret Access Key: <your AWS_SECRET_ACCESS_KEY>
# Default region: ap-south-1
# Default output format: json
```

### 5.6 Install Nginx

```bash
sudo apt-get install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

### 5.7 Create App Directory

```bash
mkdir -p ~/meetingmind
cd ~/meetingmind
```

---

## Phase 6: Configure Application Files on EC2

You need to upload two files to the EC2 server: the production compose file and the `.env`.

### 6.1 Upload docker-compose.prod.yml

From your **Windows machine** (PowerShell):

```powershell
scp -i "C:\Users\Varsha Parmar\.ssh\meetingmind-key.pem" `
    "C:\Users\Varsha Parmar\My-Projects\Fiverr-SAAS\Meeting-MasterMind-AI\docker-compose.prod.yml" `
    ubuntu@<YOUR_ELASTIC_IP>:~/meetingmind/docker-compose.prod.yml
```

### 6.2 Create Production .env on EC2

SSH into EC2 and create the env file:

```bash
nano ~/meetingmind/.env
```

Paste this template and fill in ALL values:

```env
# ─── App ───────────────────────────────────────────────
PROJECT_NAME="MeetingMind AI"
ENVIRONMENT="production"
DEBUG=false

# ─── Security ──────────────────────────────────────────
# Generate with: openssl rand -hex 32
SECRET_KEY="REPLACE_WITH_64_CHAR_HEX_STRING"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_MINUTES=10080

# ─── Database ──────────────────────────────────────────
# These values must match the docker-compose.prod.yml db service
POSTGRES_SERVER=db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=REPLACE_WITH_STRONG_PASSWORD
POSTGRES_DB=meetingmind
POSTGRES_PORT=5432

# ─── AWS ───────────────────────────────────────────────
AWS_ACCESS_KEY_ID=REPLACE
AWS_SECRET_ACCESS_KEY=REPLACE
AWS_REGION=ap-south-1
AWS_S3_BUCKET_NAME=meetingmind-meetings-YOURNAME

# ─── Transcription ─────────────────────────────────────
TRANSCRIPTION_SERVICE=aws_transcribe
AWS_TRANSCRIBE_LANGUAGE_CODE=en-US
AWS_TRANSCRIBE_OUTPUT_PREFIX=transcribe-output

# ─── LLM ───────────────────────────────────────────────
LLM_PROVIDER=claude
LLM_MODEL=claude-3-sonnet-20240229
LLM_API_KEY=REPLACE_WITH_CLAUDE_KEY
CLAUDE_API_KEY=REPLACE_WITH_CLAUDE_KEY
CLAUDE_MODEL=claude-3-sonnet-20240229

# ─── Stripe ────────────────────────────────────────────
STRIPE_SECRET_KEY=REPLACE_WITH_STRIPE_KEY
STRIPE_WEBHOOK_SECRET=REPLACE_WITH_WEBHOOK_SECRET
STRIPE_PRO_PRICE_ID=REPLACE
STRIPE_BUSINESS_PRICE_ID=REPLACE

# ─── Storage ───────────────────────────────────────────
USE_LOCAL_STORAGE=false
USE_LOCAL_WHISPER_FOR_DEV=false

# ─── Cost Tracking ─────────────────────────────────────
ENABLE_COST_TRACKING=true
MONTHLY_BUDGET_LIMIT=50.0

# ─── CORS ──────────────────────────────────────────────
# Replace with your Vercel frontend URL after deploying
CORS_ORIGINS=["https://your-app.vercel.app","http://localhost:3000"]
```

Save with Ctrl+O, Enter, Ctrl+X.

### 6.3 Also Create ECR Deploy Env File

```bash
nano ~/meetingmind/.env.deploy
```

```env
ECR_REGISTRY=123456789.dkr.ecr.ap-south-1.amazonaws.com
ECR_REPOSITORY=meetingmind-backend
IMAGE_TAG=latest
POSTGRES_PASSWORD=SAME_AS_IN_ENV_ABOVE
```

---

## Phase 7: Build and Push Docker Image

You have two options. Choose one.

### Option A: Push to ECR from your Windows machine (recommended for CI/CD)

From your Windows machine (requires Docker Desktop + AWS CLI installed locally):

```powershell
# 1. Log into ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.ap-south-1.amazonaws.com

# 2. Build the image (from the project root)
cd "C:\Users\Varsha Parmar\My-Projects\Fiverr-SAAS\Meeting-MasterMind-AI"
docker build -t meetingmind-backend ./backend

# 3. Tag it for ECR
docker tag meetingmind-backend:latest 123456789.dkr.ecr.ap-south-1.amazonaws.com/meetingmind-backend:latest

# 4. Push (this will take several minutes — ~800 MB)
docker push 123456789.dkr.ecr.ap-south-1.amazonaws.com/meetingmind-backend:latest
```

### Option B: Build directly on EC2 (avoids ECR storage cost — free tier friendly)

```bash
# On EC2 — clone only the backend code or use git (if repo is public)
# If private, use GitHub deploy key or copy files via scp

# Copy backend source to EC2 from Windows:
# (run this from PowerShell on Windows)
scp -r -i "C:\Users\Varsha Parmar\.ssh\meetingmind-key.pem" `
    "C:\Users\Varsha Parmar\My-Projects\Fiverr-SAAS\Meeting-MasterMind-AI\backend" `
    ubuntu@<YOUR_ELASTIC_IP>:~/meetingmind/backend

# Then on EC2:
cd ~/meetingmind
docker build -t meetingmind-backend ./backend
```

If you use Option B, modify `docker-compose.prod.yml` on the server to use a local image name instead of ECR:
```yaml
# Change this line in docker-compose.prod.yml:
image: ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG:-latest}
# To:
image: meetingmind-backend:latest
```

---

## Phase 8: First Production Deploy

SSH into EC2:

```bash
cd ~/meetingmind

# If using ECR (Option A), log in first:
aws ecr get-login-password --region ap-south-1 \
  | docker login --username AWS --password-stdin \
    123456789.dkr.ecr.ap-south-1.amazonaws.com

# Load the deploy env vars
export $(grep -v '^#' .env.deploy | xargs)

# Pull image and start all services
docker compose -f docker-compose.prod.yml --env-file .env.deploy up -d

# Check that both containers are running
docker compose -f docker-compose.prod.yml ps

# Run database migrations (first time only)
docker compose -f docker-compose.prod.yml exec backend alembic upgrade head

# Verify the API is responding
curl http://localhost:8000/health
```

Expected response: `{"status":"healthy"}` or similar.

---

## Phase 9: Nginx Reverse Proxy + HTTPS

### 9.1 Create Nginx Config

```bash
sudo nano /etc/nginx/sites-available/meetingmind
```

Paste:

```nginx
server {
    listen 80;
    server_name <YOUR_ELASTIC_IP>;  # replace with domain if you have one

    # Max upload size for meeting files (100 MB)
    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # Long timeout for meeting upload + processing
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/meetingmind /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t        # should say: syntax is ok
sudo systemctl reload nginx

# Test
curl http://<YOUR_ELASTIC_IP>/health
```

### 9.2 HTTPS with Let's Encrypt (requires a domain name)

If you have a domain pointed to your Elastic IP:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
# Follow prompts — enter vparmarce@gmail.com when asked for email
# Choose option 2 (Redirect HTTP to HTTPS)
```

Certbot auto-renews every 90 days via a cron job it installs itself.

If you don't have a domain yet, skip this step and use the IP address directly for now.

### 9.3 Remove Port 8000 from Security Group

Now that Nginx handles routing, close the direct port:

1. EC2 Console → Security Groups → `meetingmind-sg` → Inbound rules
2. Delete the rule for port 8000
3. Only ports 22, 80, and 443 should remain

---

## Phase 10: Frontend Deployment on Vercel

Vercel is free for Next.js and is the easiest way to deploy your frontend.

### 10.1 Push Frontend to GitHub

Make sure your monorepo is pushed to GitHub (it should already be).

### 10.2 Deploy on Vercel

1. Go to https://vercel.com → **Add New Project**
2. Import your GitHub repository (`My-Projects` or whatever the repo is called)
3. **Root Directory:** set to `Fiverr-SAAS/Meeting-MasterMind-AI/frontend`
4. Framework: **Next.js** (auto-detected)
5. **Environment Variables** — add these:

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_URL` | `http://<YOUR_ELASTIC_IP>` (or `https://yourdomain.com` if HTTPS) |

6. Click **Deploy**

### 10.3 Update CORS on Backend

After Vercel gives you a URL (e.g. `https://meetingmind-abc.vercel.app`), update the `.env` on EC2:

```bash
nano ~/meetingmind/.env
# Update CORS_ORIGINS:
CORS_ORIGINS=["https://meetingmind-abc.vercel.app","http://localhost:3000"]
```

Then restart the backend:

```bash
cd ~/meetingmind
docker compose -f docker-compose.prod.yml restart backend
```

Also update the S3 CORS policy to add the Vercel URL (Phase 2.2).

---

## Phase 11: Configure GitHub Secrets for CI/CD

Your CI/CD workflow at `.github/workflows/meetingmind-ci-cd.yml` needs these secrets to auto-deploy on push to `main`.

GitHub → Your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Secret Name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | Your IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM user secret key |
| `AWS_REGION` | `ap-south-1` |
| `ECR_REPOSITORY` | `meetingmind-backend` |
| `ECR_REGISTRY` | `123456789.dkr.ecr.ap-south-1.amazonaws.com` |
| `EC2_HOST` | Your Elastic IP address |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | Full content of `meetingmind-key.pem` (copy-paste the entire file including `-----BEGIN RSA PRIVATE KEY-----` and `-----END RSA PRIVATE KEY-----`) |

Once these are set, every push to `main` that touches `Fiverr-SAAS/Meeting-MasterMind-AI/**` will:
1. Run `ruff` lint
2. Run 90 pytest tests
3. Build + push Docker image to ECR
4. SSH into EC2 and restart the backend container

---

## Phase 12: Verify End-to-End

### 12.1 Backend Health Check

```bash
curl https://yourdomain.com/health
# or
curl http://<YOUR_ELASTIC_IP>/health
```

### 12.2 API Docs (temporarily)

In production, Swagger is disabled. To check endpoints in dev mode, temporarily set `DEBUG=true` in `.env` and restart, then visit `http://<YOUR_ELASTIC_IP>/docs`.

### 12.3 Register a User via Frontend

1. Open your Vercel URL
2. Register a new account
3. Create a workspace
4. Upload a short `.wav` file (use `backend/sample_meeting.wav` for testing)
5. Wait 1-5 minutes for transcription via AWS Transcribe
6. Check the meeting detail page for transcript + summary

---

## Phase 13: Monitoring & Logs

### View Live Logs

```bash
# Backend logs (tail -f equivalent)
docker compose -f ~/meetingmind/docker-compose.prod.yml logs -f backend

# DB logs
docker compose -f ~/meetingmind/docker-compose.prod.yml logs -f db

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Check Container Status

```bash
docker compose -f ~/meetingmind/docker-compose.prod.yml ps
docker stats   # live CPU/RAM usage
```

### Set Up CloudWatch Basic Monitoring (free)

EC2 already pushes basic CPU/network metrics to CloudWatch for free.

1. EC2 Console → your instance → **Monitoring** tab
2. **Enable detailed monitoring**: NO (detailed is paid; basic is free)
3. Create a CloudWatch Alarm:
   - Metric: CPUUtilization > 80% for 2 consecutive periods
   - Action: Send email to vparmarce@gmail.com
   - This prevents runaway processing from maxing out the instance

---

## Phase 14: Manual Re-deploy Steps (for updates without CI/CD)

If you want to manually redeploy after a code change:

### On your Windows machine:

```powershell
# Rebuild image
cd "C:\Users\Varsha Parmar\My-Projects\Fiverr-SAAS\Meeting-MasterMind-AI"
docker build -t meetingmind-backend ./backend

# Re-tag and push to ECR
docker tag meetingmind-backend:latest 123456789.dkr.ecr.ap-south-1.amazonaws.com/meetingmind-backend:latest
docker push 123456789.dkr.ecr.ap-south-1.amazonaws.com/meetingmind-backend:latest
```

### On EC2 (SSH in):

```bash
cd ~/meetingmind

# Pull latest image
aws ecr get-login-password --region ap-south-1 \
  | docker login --username AWS --password-stdin \
    123456789.dkr.ecr.ap-south-1.amazonaws.com

export $(grep -v '^#' .env.deploy | xargs)
docker compose -f docker-compose.prod.yml pull backend

# Restart with zero-downtime (only backend, db keeps running)
docker compose -f docker-compose.prod.yml up -d --no-deps backend

# Run any pending migrations
docker compose -f docker-compose.prod.yml exec backend alembic upgrade head
```

---

## Troubleshooting

### Container won't start

```bash
docker compose -f ~/meetingmind/docker-compose.prod.yml logs backend
```

Most common causes:
- Missing env var → check `.env` has all required keys
- DB not ready → wait 10 seconds and try again (healthcheck retries handle this)
- Port already in use → `sudo lsof -i :8000`

### Out of disk space (t2.micro has limited storage)

```bash
df -h               # check disk usage
docker system prune # clean up old images/containers (careful: removes stopped containers)
docker image ls     # see what images exist
docker image rm <id>  # remove old images manually
```

### t2.micro runs out of RAM (1 GB is tight)

The backend container + PostgreSQL together can approach 700-800 MB RAM. If you see OOM kills:

```bash
# Check memory
free -h

# Add a swap file (virtual memory) — free, uses disk space
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Make it permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### AWS Transcribe not working

Check IAM permissions. The `meetingmind-deploy` user needs:
```
transcribe:StartTranscriptionJob
transcribe:GetTranscriptionJob
```

Add via IAM → Users → meetingmind-deploy → Add permissions → Attach `AmazonTranscribeFullAccess`.

### Database connection errors

```bash
# Check if db container is healthy
docker compose -f ~/meetingmind/docker-compose.prod.yml ps
# Should show "healthy" for db

# Connect manually to verify
docker compose -f ~/meetingmind/docker-compose.prod.yml exec db \
  psql -U postgres -d meetingmind -c "\dt"
```

---

## Free Tier Monthly Cost Estimate

| Service | Free | Pay | Your Usage |
|---------|------|-----|-----------|
| EC2 t2.micro | 750 hrs = 31 days | $0 | Always running |
| EBS 20 GB | 30 GB free | $0 | Under limit |
| Elastic IP (attached) | Free | $0 | Attached to running instance |
| S3 storage | 5 GB | $0.023/GB | < 5 GB for testing |
| S3 requests | 20K GET, 2K PUT | ~$0 | Low volume |
| ECR | 500 MB | ~$0.03/GB/month | 800 MB image → ~$0.03/month |
| AWS Transcribe | 60 min/month | $0.024/min | < 60 min testing |
| Data transfer out | 1 GB/month | $0.09/GB | Low volume |
| **Total (testing)** | | **~$0–$1/month** | |

> After 12 months, free tier expires. EC2 t2.micro becomes ~$8.50/month. Consider upgrading to t3.micro ($7.59/month) which is also free tier eligible and has burst CPU.

---

## Quick Reference: Key Commands

```bash
# SSH into server
ssh -i "C:\Users\Varsha Parmar\.ssh\meetingmind-key.pem" ubuntu@<ELASTIC_IP>

# View logs
docker compose -f ~/meetingmind/docker-compose.prod.yml logs -f backend

# Restart backend
docker compose -f ~/meetingmind/docker-compose.prod.yml restart backend

# Run migrations
docker compose -f ~/meetingmind/docker-compose.prod.yml exec backend alembic upgrade head

# Full restart (all services)
docker compose -f ~/meetingmind/docker-compose.prod.yml down
docker compose -f ~/meetingmind/docker-compose.prod.yml up -d

# Check disk space
df -h

# Check RAM
free -h

# Check running containers
docker compose -f ~/meetingmind/docker-compose.prod.yml ps
```

---

## Checklist

- [ ] Phase 1: AWS account + billing alert + IAM user created
- [ ] Phase 2: S3 bucket created with CORS policy
- [ ] Phase 3: ECR repository created
- [ ] Phase 4: EC2 t2.micro launched + Elastic IP attached + SSH works
- [ ] Phase 5: Docker, Docker Compose, AWS CLI, Nginx installed on EC2
- [ ] Phase 6: `docker-compose.prod.yml` and `.env` uploaded to `~/meetingmind/`
- [ ] Phase 7: Docker image built and pushed (ECR or local)
- [ ] Phase 8: `docker compose up -d` succeeds + `alembic upgrade head` runs + `/health` returns 200
- [ ] Phase 9: Nginx configured + port 8000 removed from security group
- [ ] Phase 10: Frontend deployed on Vercel + `NEXT_PUBLIC_API_URL` set + CORS updated
- [ ] Phase 11: GitHub Secrets configured (8 secrets) + CI/CD pipeline test-pushes successfully
- [ ] Phase 12: End-to-end test: register → upload meeting → get transcript
- [ ] Phase 13: CloudWatch CPU alarm set
