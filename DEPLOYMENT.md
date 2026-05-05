# 🚀 OmniDetector Deployment Guide

Deploy your OmniDetector app online for 24/7 availability across multiple platforms.

## Quick Start Options

### Option 1: Hugging Face Spaces (Recommended - Free 24/7)
**Best for: Free deployment with no sleep issues**

1. **Prerequisites**
   - GitHub account with your repository
   - Hugging Face account (free)

2. **Deploy Steps**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" as SDK  
   - Connect your GitHub repository
   - The `.spaces.yml` file will automatically configure everything

3. **Configuration**
   - Space will use the `README_HF.md` as description
   - Models download automatically on first run
   - No manual configuration needed

4. **Access**: Available at `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

**Advantages:**
- ✅ Completely free forever
- ✅ No sleep/pause issues (24/7 availability)
- ✅ Automatic HTTPS and SSL
- ✅ Built-in CI/CD from GitHub
- ✅ Optimized for ML applications
- ✅ Community visibility and sharing

---

### Option 2: Railway (Best Paid Option - $5/month)
**Best for: Professional deployment with guaranteed uptime**

1. **Setup**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login and deploy
   railway login
   railway init
   railway up
   ```

2. **Configuration**
   - Uses `railway.json` for automatic configuration
   - Dockerfile deployment with health checks
   - Automatic domain and HTTPS

3. **Pricing**: $5/month for 500 hours (more than enough for 24/7)

**Advantages:**
- ✅ Guaranteed 24/7 uptime
- ✅ Fast deployment and scaling
- ✅ Professional custom domains
- ✅ Advanced monitoring and logs
- ✅ Excellent performance

---

### Option 3: Render (Free with Limitations)
**Best for: Free deployment with occasional sleep**

1. **Deploy from GitHub**
   - Connect your GitHub repository
   - Uses `render.yaml` for configuration
   - Automatic builds on code changes

2. **Limitations**
   - Free tier sleeps after 15 minutes of inactivity
   - Takes ~30 seconds to wake up from sleep
   - 750 hours/month limit (not 24/7)

**Advantages:**
- ✅ Free tier available
- ✅ Easy GitHub integration
- ✅ Automatic SSL certificates
- ❌ Sleep issues on free tier

---

### Option 4: Docker Deployment (Self-Hosted)
**Best for: Full control and self-hosting**

1. **Using Docker Compose** (Recommended)
   ```bash
   # Clone repository
   git clone YOUR_REPO_URL
   cd OmniDetector
   
   # Build and run
   docker-compose up -d
   ```

2. **Manual Docker Build**
   ```bash
   # Build image
   docker build -t omnidetector .
   
   # Run container
   docker run -d -p 8501:8501 --name omnidetector omnidetector
   ```

3. **Access**: http://localhost:8501

**Advantages:**
- ✅ Full control over hosting
- ✅ No platform limitations
- ✅ Can run on any server
- ✅ Custom domain configuration

---

## Platform Comparison

| Platform | Cost | 24/7 Uptime | Sleep Issues | Setup Difficulty | Performance |
|----------|------|--------------|--------------|------------------|-------------|
| **Hugging Face Spaces** | Free | ✅ Yes | ❌ None | ⭐ Easy | ⭐⭐⭐⭐ |
| **Railway** | $5/month | ✅ Yes | ❌ None | ⭐⭐ Moderate | ⭐⭐⭐⭐⭐ |
| **Render** | Free/Paid | ❌ No* | ⚠️ 15min idle | ⭐ Easy | ⭐⭐⭐ |
| **Docker Self-Host** | Server cost | ✅ Yes | ❌ None | ⭐⭐⭐ Advanced | ⭐⭐⭐⭐⭐ |

*Render free tier has sleep issues

---

## Detailed Setup Instructions

### Hugging Face Spaces Detailed Steps

1. **Prepare Repository**
   - Ensure your repository is public on GitHub
   - Verify `README_HF.md` and `.spaces.yml` files are present
   - Check that `requirements.txt` is up to date

2. **Create Space**
   - Visit https://huggingface.co/spaces
   - Click "Create new Space"
   - Fill in details:
     - Space name: `omnidetector-ultimate`
     - License: `mit`
     - SDK: `Streamlit`
     - Hardware: `CPU basic` (free)

3. **Connect Repository**
   - Link your GitHub repository
   - Set branch to `main` or `master`
   - Enable automatic rebuilds

4. **Wait for Build**
   - Initial build takes 5-10 minutes
   - Models download automatically on first run
   - Check build logs for any issues

5. **Access Your App**
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/omnidetector-ultimate`
   - Share with anyone - no login required for users

### Railway Detailed Steps

1. **Install Railway CLI**
   ```bash
   # Windows (with Node.js)
   npm install -g @railway/cli
   
   # macOS
   brew install railway/tap/railway
   
   # Linux
   curl -sSL https://railway.app/install.sh | sh
   ```

2. **Deploy Project**
   ```bash
   # Navigate to your project
   cd path/to/OmniDetector
   
   # Login to Railway
   railway login
   
   # Initialize project
   railway init
   
   # Deploy
   railway up
   ```

3. **Configure Domain**
   - Visit Railway dashboard
   - Go to your project settings
   - Configure custom domain (optional)
   - SSL certificates are automatic

4. **Monitor Deployment**
   - Check deployment logs
   - Set up monitoring alerts
   - Configure environment variables if needed

### Render Detailed Steps

1. **Connect GitHub**
   - Go to https://render.com
   - Click "New" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   - Name: `omnidetector-ultimate`
   - Environment: `Docker`
   - Branch: `main`
   - Build and deploy automatically

3. **Environment Variables** (if needed)
   ```
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   ```

4. **Monitor Deployment**
   - Check build logs
   - Test application functionality
   - Note sleep behavior on free tier

---

## Troubleshooting

### Common Issues

**1. Build Failures**
- Check `requirements.txt` for compatibility
- Verify Python version (3.9+ recommended)
- Check Docker build logs

**2. Model Download Issues**
- Ensure internet connectivity during build
- Check if models directory is writable
- Verify model URLs are accessible

**3. Memory Issues**
- Use smaller YOLO models (YOLOv8n instead of YOLOv8m)
- Optimize image processing batch sizes
- Check platform memory limits

**4. Port Configuration**
- Streamlit default: 8501
- Railway: Uses $PORT environment variable
- Docker: Expose port 8501

### Platform-Specific Issues

**Hugging Face Spaces**
- Build timeout: Use lighter models
- Space sleeping: Shouldn't happen, contact support
- Authentication: Not needed for public spaces

**Railway**
- Build failures: Check Dockerfile syntax
- Domain issues: Verify DNS settings
- Billing: Monitor usage to avoid overages

**Render**
- Sleep issues: Upgrade to paid tier
- Build timeout: Optimize build process
- Memory limits: Use efficient model loading

---

## Performance Optimization

### For Cloud Deployment

1. **Model Selection**
   ```python
   # Use lighter models for faster loading
   model_options = {
       'YOLOv8n': 'yolov8n.pt',  # Fastest, less accurate
       'YOLOv8s': 'yolov8s.pt',  # Balanced
       'YOLOv8m': 'yolov8m.pt'   # Slower, more accurate
   }
   ```

2. **Caching Strategy**
   ```python
   @st.cache_resource
   def load_model(model_path):
       return YOLO(model_path)
   ```

3. **Memory Management**
   - Limit image sizes
   - Clear variables after processing
   - Use efficient data structures

### For Self-Hosted

1. **Docker Optimization**
   ```dockerfile
   # Multi-stage builds for smaller images
   FROM python:3.9-slim as builder
   # ... build dependencies
   
   FROM python:3.9-slim as runtime
   # ... runtime only
   ```

2. **Resource Limits**
   ```yaml
   # docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 2G
         cpus: '1.0'
   ```

---

## Security Considerations

### Production Deployment

1. **Environment Variables**
   ```bash
   # Don't hardcode secrets
   export STREAMLIT_SERVER_HEADLESS=true
   export MODEL_CACHE_DIR=/app/models
   ```

2. **Docker Security**
   ```dockerfile
   # Run as non-root user
   RUN adduser --disabled-password --gecos '' appuser
   USER appuser
   ```

3. **HTTPS Configuration**
   - All platforms provide automatic HTTPS
   - For self-hosted: Use reverse proxy (nginx, caddy)

### Privacy and Data

- No user data is stored permanently
- Images processed in memory only
- Models downloaded from official sources
- No tracking or analytics collection

---

## Support and Maintenance

### Getting Help

1. **Platform Support**
   - Hugging Face: Community forums and documentation
   - Railway: Discord community and docs
   - Render: Support tickets and documentation

2. **Repository Issues**
   - Check GitHub issues for common problems
   - Create new issues for bugs or features
   - Contribute improvements via pull requests

### Updates and Maintenance

1. **Automated Updates**
   - Enable automatic rebuilds on GitHub pushes
   - Monitor for security updates
   - Update dependencies regularly

2. **Manual Maintenance**
   - Check model performance periodically
   - Update YOLO models when new versions release
   - Monitor resource usage and costs

---

## Conclusion

For **4-5 months of 24/7 hosting without sleep issues**, the best options are:

1. **🥇 Hugging Face Spaces** - Completely free, no sleep issues, perfect for ML apps
2. **🥈 Railway** - $5/month but guaranteed uptime and professional features  
3. **🥉 Docker Self-Host** - Full control but requires server management

**Recommendation**: Start with Hugging Face Spaces for free 24/7 hosting, then consider Railway if you need additional features or custom domains.

All deployment files are included in this repository:
- `.spaces.yml` - Hugging Face Spaces configuration
- `railway.json` - Railway deployment configuration  
- `render.yaml` - Render deployment configuration
- `Dockerfile` - Container deployment
- `docker-compose.yml` - Local Docker deployment

Choose the platform that best fits your needs and budget!