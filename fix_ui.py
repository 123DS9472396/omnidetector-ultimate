import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('<h1 class="main-title">🚀 OmniDetector Ultimate v3.0</h1>', '<h2 class="main-title" style="font-size: 2.2rem; margin-bottom: 0.5rem; text-shadow: 0 2px 10px rgba(0,0,0,0.3);">🚀 OmniDetector Ultimate v3.0</h2>')
content = content.replace('<p class="subtitle">World\'s Most Advanced Real-Time Object Detection System</p>', '<p class="subtitle" style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 1rem;">World\'s Most Advanced Real-Time Object Detection System</p>')
content = content.replace('<div class="header-container">', '<div class="header-container" style="padding: 1.5rem; margin-bottom: 1rem;">')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
