// StoxAI - Neural Network Background Animation
// Runs when DOM is ready (handles both late script load and normal load)

function initNeuralBackground() {
    const canvas = document.getElementById('neural-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let width = window.innerWidth;
    let height = window.innerHeight;
    
    canvas.width = width;
    canvas.height = height;
    
    // Neural network configuration
    const config = {
        nodeCount: 80,
        connectionDistance: 150,
        nodeSpeed: 0.3,
        nodeSize: 2,
        primaryColor: 'rgba(29, 185, 84, 0.8)',
        secondaryColor: 'rgba(16, 163, 74, 0.6)',
        lineColor: 'rgba(29, 185, 84, 0.15)',
        glowColor: 'rgba(29, 185, 84, 0.3)',
    };
    
    // Node class
    class Node {
        constructor() {
            this.reset();
        }
        
        reset() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * config.nodeSpeed;
            this.vy = (Math.random() - 0.5) * config.nodeSpeed;
            this.radius = config.nodeSize + Math.random() * 2;
        }
        
        update() {
            this.x += this.vx;
            this.y += this.vy;
            
            // Bounce off edges
            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;
            
            // Keep in bounds
            this.x = Math.max(0, Math.min(width, this.x));
            this.y = Math.max(0, Math.min(height, this.y));
        }
        
        draw() {
            // Draw glow
            ctx.beginPath();
            const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius * 3);
            gradient.addColorStop(0, config.glowColor);
            gradient.addColorStop(1, 'rgba(29, 185, 84, 0)');
            ctx.fillStyle = gradient;
            ctx.arc(this.x, this.y, this.radius * 3, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw node
            ctx.beginPath();
            ctx.fillStyle = config.primaryColor;
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    // Create nodes
    const nodes = [];
    for (let i = 0; i < config.nodeCount; i++) {
        nodes.push(new Node());
    }
    
    // Draw connections
    function drawConnections() {
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < config.connectionDistance) {
                    const opacity = 1 - (distance / config.connectionDistance);
                    ctx.beginPath();
                    ctx.strokeStyle = config.lineColor.replace('0.15', opacity * 0.15);
                    ctx.lineWidth = opacity * 1.5;
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.stroke();
                    
                    // Draw data pulse
                    if (Math.random() > 0.99) {
                        const pulseX = nodes[i].x + (dx * Math.random());
                        const pulseY = nodes[i].y + (dy * Math.random());
                        
                        ctx.beginPath();
                        const pulseGradient = ctx.createRadialGradient(pulseX, pulseY, 0, pulseX, pulseY, 4);
                        pulseGradient.addColorStop(0, config.primaryColor);
                        pulseGradient.addColorStop(1, 'rgba(29, 185, 84, 0)');
                        ctx.fillStyle = pulseGradient;
                        ctx.arc(pulseX, pulseY, 4, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            }
        }
    }
    
    // Animation loop
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        // Update and draw nodes
        nodes.forEach(node => {
            node.update();
            node.draw();
        });
        
        // Draw connections
        drawConnections();
        
        requestAnimationFrame(animate);
    }
    
    // Handle resize
    window.addEventListener('resize', function() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
        
        // Reset nodes to new dimensions
        nodes.forEach(node => node.reset());
    });
    
    // Start animation
    animate();
    
    // Show/hide loading overlay
    window.showLoading = function() {
        const loader = document.getElementById('global-loading');
        if (loader) {
            loader.style.display = 'flex';
            setTimeout(() => {
                loader.style.opacity = '1';
            }, 10);
        }
    };
    
    window.hideLoading = function() {
        const loader = document.getElementById('global-loading');
        if (loader) {
            loader.style.opacity = '0';
            setTimeout(() => {
                loader.style.display = 'none';
            }, 300);
        }
    };
    
    // Auto-hide loading on page load
    setTimeout(() => {
        window.hideLoading();
    }, 1000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNeuralBackground);
} else {
    initNeuralBackground();
}