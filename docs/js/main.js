/**
 * NI6 Main JavaScript
 * Handles AOS initialization, typewriter effect, and interactive features
 */

// Initialize AOS (Animate On Scroll)
function initAOS() {
    AOS.init({
        duration: 800,
        easing: 'ease-out-cubic',
        once: true,
        offset: 50,
        delay: 0,
    });
}

// Typewriter Effect for Hero Tagline
function initTypewriter() {
    const taglineElement = document.getElementById('typewriter');
    if (!taglineElement) return;

    const phrases = [
        'Algorithmic Consistency.',
        'Verifiable Neutrality.',
        'Decentralized Custody.',
        'Gesture Based Recovery.',
        'Zero API Costs.',
        'Privacy by Design.'
    ];

    let phraseIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    let typingSpeed = 100;

    function type() {
        const currentPhrase = phrases[phraseIndex];

        if (isDeleting) {
            // Deleting characters
            taglineElement.textContent = currentPhrase.substring(0, charIndex - 1);
            charIndex--;
            typingSpeed = 50; // Faster when deleting
        } else {
            // Typing characters
            taglineElement.textContent = currentPhrase.substring(0, charIndex + 1);
            charIndex++;
            typingSpeed = 100 + Math.random() * 50; // Natural typing variation
        }

        // Check if phrase is complete
        if (!isDeleting && charIndex === currentPhrase.length) {
            // Pause at end of phrase
            typingSpeed = 2000;
            isDeleting = true;
        } else if (isDeleting && charIndex === 0) {
            // Move to next phrase
            isDeleting = false;
            phraseIndex = (phraseIndex + 1) % phrases.length;
            typingSpeed = 500;
        }

        setTimeout(type, typingSpeed);
    }

    // Start typing after a short delay
    setTimeout(type, 1000);
}

// Copy to Clipboard Functionality
function initCopyButton() {
    const copyBtn = document.getElementById('copyBtn');
    const installCode = document.getElementById('installCode');

    if (!copyBtn || !installCode) return;

    copyBtn.addEventListener('click', async () => {
        const code = installCode.textContent;

        try {
            await navigator.clipboard.writeText(code);
            copyBtn.classList.add('copied');
            copyBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
                Copied!
            `;

            // Reset button after 2 seconds
            setTimeout(() => {
                copyBtn.classList.remove('copied');
                copyBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                    Copy
                `;
            }, 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
            // Fallback for older browsers
            fallbackCopy(code);
        }
    });
}

// Fallback copy method for older browsers
function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    document.body.appendChild(textArea);
    textArea.select();

    try {
        document.execCommand('copy');
        const copyBtn = document.getElementById('copyBtn');
        copyBtn.classList.add('copied');
        copyBtn.textContent = 'Copied!';
        setTimeout(() => {
            copyBtn.classList.remove('copied');
            copyBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                Copy
            `;
        }, 2000);
    } catch (err) {
        console.error('Fallback copy failed:', err);
    }

    document.body.removeChild(textArea);
}

// Smooth scroll for navigation links
function initSmoothScroll() {
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href === '#') return;

            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Navbar background on scroll
function initNavbarScroll() {
    const nav = document.querySelector('.nav');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 50) {
            nav.style.background = 'rgba(10, 10, 15, 0.95)';
        } else {
            nav.style.background = 'rgba(10, 10, 15, 0.8)';
        }

        lastScroll = currentScroll;
    });
}

// Intersection Observer for table row highlights
function initTableHighlights() {
    const tableRows = document.querySelectorAll('.table-row');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Stagger animation for each row
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateX(0)';
                }, index * 100);
            }
        });
    }, {
        threshold: 0.2,
        rootMargin: '0px 0px -50px 0px'
    });

    tableRows.forEach(row => {
        row.style.opacity = '0';
        row.style.transform = 'translateX(-20px)';
        row.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(row);
    });
}

// Parallax effect for hero background grid
function initParallax() {
    const heroBg = document.querySelector('.hero-bg .grid');
    if (!heroBg) return;

    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const parallaxSpeed = 0.3;

        if (scrolled < window.innerHeight) {
            heroBg.style.transform = `translateY(${scrolled * parallaxSpeed}px)`;
        }
    });
}

// Tech item hover animation
function initTechAnimations() {
    const techItems = document.querySelectorAll('.tech-item');

    techItems.forEach(item => {
        item.addEventListener('mouseenter', () => {
            // Add a subtle glow effect
            item.style.boxShadow = '0 0 20px rgba(99, 102, 241, 0.3)';
        });

        item.addEventListener('mouseleave', () => {
            item.style.boxShadow = 'none';
        });
    });
}

// Add counter animation for metrics in floating card
function initMetricCounters() {
    const metricValues = document.querySelectorAll('.metric-value');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;
                const finalValue = target.textContent;

                // Only animate numeric values
                if (/[\d.]+%/.test(finalValue)) {
                    const numericValue = parseFloat(finalValue);
                    animateValue(target, 0, numericValue, 1500, '%');
                } else if (/\d+\/\d+/.test(finalValue)) {
                    // For scores like "15/100", just show the value
                    target.style.opacity = '0';
                    setTimeout(() => {
                        target.style.transition = 'opacity 0.5s';
                        target.style.opacity = '1';
                    }, 100);
                }

                observer.unobserve(target);
            }
        });
    }, { threshold: 0.5 });

    metricValues.forEach(value => observer.observe(value));
}

function animateValue(element, start, end, duration, suffix = '') {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if (current >= end) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = current.toFixed(1) + suffix;
    }, 16);
}

// Detect reduced motion preference
function respectReducedMotion() {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

    if (prefersReducedMotion.matches) {
        // Disable AOS animations
        if (typeof AOS !== 'undefined') {
            AOS.init({
                disable: true
            });
        }

        // Disable typewriter effect
        const taglineElement = document.getElementById('typewriter');
        if (taglineElement) {
            taglineElement.textContent = 'Algorithmic Consistency. Verifiable Neutrality.';
        }

        // Disable floating animation
        const floatingCard = document.querySelector('.floating-card');
        if (floatingCard) {
            floatingCard.style.animation = 'none';
        }

        // Show static gesture (gesture_2) instead of animating
        const gestureImage = document.getElementById('gesture-animation');
        if (gestureImage) {
            gestureImage.setAttribute('href', 'assets/images/gesture_2.png');
        }
    }
}

// Initialize everything when DOM is ready
function init() {
    // Check for reduced motion preference
    respectReducedMotion();

    // Initialize AOS
    initAOS();

    // Initialize features
    initTypewriter();
    initCopyButton();
    initSmoothScroll();
    initNavbarScroll();
    initParallax();
    initTechAnimations();
    initMetricCounters();
    initGestureAnimation();

    // Initialize table highlights after AOS animations complete
    setTimeout(initTableHighlights, 1000);

    // Initialize syntax highlighting
    if (typeof hljs !== 'undefined') {
        hljs.highlightAll();
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Re-initialize AOS on window resize (for responsive adjustments)
let resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        if (typeof AOS !== 'undefined') {
            AOS.refresh();
        }
    }, 250);
});

// Add active state to navigation based on scroll position
function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');

    let currentSection = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        const sectionHeight = section.offsetHeight;

        if (window.pageYOffset >= sectionTop && window.pageYOffset < sectionTop + sectionHeight) {
            currentSection = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.style.color = '';
        if (link.getAttribute('href') === `#${currentSection}`) {
            link.style.color = 'var(--color-primary)';
        }
    });
}

window.addEventListener('scroll', updateActiveNavLink);

// Gesture Animation - cycles through gesture images 1-5
function initGestureAnimation() {
    const gestureImage = document.getElementById('gesture-image');
    if (!gestureImage) {
        console.error('Gesture animation element NOT FOUND!');
        return;
    }

    console.log('Gesture animation initialized!');
    console.log('Element:', gestureImage);
    console.log('Initial src:', gestureImage.src);
    console.log('Display:', window.getComputedStyle(gestureImage).display);

    // Test image load
    gestureImage.onload = function() {
        console.log('Gesture image loaded successfully!');
    };
    gestureImage.onerror = function() {
        console.error('Gesture image FAILED to load!');
    };

    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (prefersReducedMotion.matches) {
        // Show static gesture
        gestureImage.src = 'assets/images/gesture_2.png';
        console.log('Reduced motion detected, showing static gesture');
        return;
    }

    const gestures = ['gesture_1', 'gesture_2', 'gesture_3', 'gesture_4', 'gesture_5', 'gesture_0'];
    let currentGesture = 0;

    setInterval(() => {
        currentGesture = (currentGesture + 1) % gestures.length;
        const newPath = `assets/images/${gestures[currentGesture]}.png`;
        console.log('Changing gesture to:', newPath);
        gestureImage.src = newPath;
    }, 1500); // Change gesture every 1.5 seconds
}

// Console Easter egg
console.log('%c NI6 ', 'background: linear-gradient(135deg, #00d4aa, #6366f1); color: #000; font-size: 20px; font-weight: bold; padding: 5px;');
console.log('%c Algorithmic Consistency. Verifiable Neutrality. ', 'color: #00d4aa; font-size: 12px;');
