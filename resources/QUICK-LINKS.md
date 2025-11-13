# CUDA Quick Links - Bookmark This Page

## 🔖 Essential URLs (Copy to Browser Bookmarks)

### Official Documentation
```
CUDA Programming Guide:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/

CUDA Best Practices:
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

All CUDA Docs:
https://docs.nvidia.com/cuda/
```

### Learning Resources
```
NVIDIA Training (FREE):
https://www.nvidia.com/en-us/training/

CUDA Samples GitHub:
https://github.com/NVIDIA/cuda-samples

Caltech CS179:
http://courses.cms.caltech.edu/cs179/

GTC On-Demand:
https://www.nvidia.com/gtc/
```

### Community & Help
```
NVIDIA Developer Forums:
https://forums.developer.nvidia.com/c/accelerated-computing/cuda/206

Stack Overflow CUDA:
https://stackoverflow.com/questions/tagged/cuda

Reddit r/CUDA:
https://reddit.com/r/CUDA
```

### Tools Documentation
```
NSight Compute:
https://docs.nvidia.com/nsight-compute/

NSight Systems:
https://docs.nvidia.com/nsight-systems/

CUDA Toolkit:
https://developer.nvidia.com/cuda-toolkit
```

### Your Local Files
```
Your Progress:
file:///home/hesham/cuda-learning/PROGRESS.md

Free Resources Guide:
file:///home/hesham/cuda-learning/resources/FREE-RESOURCES.md

Cheat Sheet:
file:///home/hesham/cuda-learning/resources/cheatsheet.md
```

---

## 📱 Quick Search Commands

### Google Searches That Work

**When stuck on error:**
```
cuda error <error_code>
```

**Looking for examples:**
```
cuda <algorithm_name> example site:github.com
```

**Official docs:**
```
cuda <topic> site:docs.nvidia.com
```

**University resources:**
```
cuda programming course site:edu
```

**Latest techniques:**
```
cuda optimization 2024
```

---

## 🎯 Workflow Quick Reference

### Daily Coding Session
1. **Before coding:** Check cheatsheet.md
2. **While stuck:** Search Stack Overflow
3. **Understanding concept:** CUDA Programming Guide
4. **Optimizing:** Best Practices Guide
5. **Inspiration:** CUDA samples

### Weekly Deep Dive
1. **Sunday:** Plan week (COMPREHENSIVE-CURRICULUM.md)
2. **Daily:** Complete lessons
3. **Thursday:** Profile code (NSight)
4. **Saturday:** Read paper/watch GTC talk
5. **Sunday:** Update PROGRESS.md

---

## 📚 Resources by Week

### Week 1
- ✓ Your lesson01-05 (PRIMARY)
- Programming Guide Ch 1-3
- "An Even Easier Introduction to CUDA" (NVIDIA blog)

### Week 2
- ✓ Your memory exercises
- Best Practices Guide: Memory chapter
- CUDA samples: memory examples

### Week 3
- Best Practices Guide: Optimization
- "Optimizing Parallel Reduction" paper
- NSight Compute documentation

### Week 4+
- Advanced GTC sessions
- Research papers
- GPU Gems chapters

---

## ⚡ Emergency Quick Help

### Compilation Error
1. Check error message
2. Search: `cuda <error> site:stackoverflow.com`
3. Check syntax in cheatsheet.md

### Runtime Error
1. Run: `cuda-memcheck ./program`
2. Add debug prints in kernel
3. Search error on forums

### Performance Issue
1. Profile: `nsys profile ./program`
2. Check: Best Practices Guide
3. Compare: CUDA samples

### Concept Confusion
1. Re-read: Your lesson
2. Check: Programming Guide
3. Watch: Related GTC talk

---

## 💾 Download These Now

### Priority Downloads

1. **CUDA Samples** (already cloned!)
   ```bash
   cd ~/cuda-learning/external-resources/cuda-samples
   ```

2. **CUDA Programming Guide PDF**
   - Visit: https://docs.nvidia.com/cuda/
   - Download PDF for offline reading

3. **Quick Reference Card**
   - Search: "CUDA quick reference card PDF"
   - Print and keep at desk

4. **Caltech Lecture Notes**
   - Download all PDFs from course site
   - Read while commuting

---

## 🎯 Your Action Items (Next 10 Minutes)

### Setup Your Browser Bookmarks

**Create folder:** "CUDA Learning"

**Add these bookmarks:**
1. CUDA Docs (main page)
2. NVIDIA Forums
3. Stack Overflow CUDA tag
4. CUDA Samples GitHub
5. Your PROGRESS.md (local file)
6. FREE-RESOURCES.md (local file)

**Browser search shortcuts:**
- Keyword: "cuda" → https://docs.nvidia.com/cuda/
- Keyword: "cuda-so" → https://stackoverflow.com/questions/tagged/cuda

---

## 📞 Who to Ask What

### General CUDA Questions
→ **Stack Overflow** (fast response, searchable)

### NVIDIA-Specific Issues
→ **NVIDIA Forums** (official support)

### Algorithm Help
→ **Reddit r/CUDA** (community discussions)

### Performance Optimization
→ **GitHub Issues** (on CUDA samples repo)

### Career/Learning Path
→ **LinkedIn CUDA Groups** (professional network)

---

## 🎓 Free Courses - Enrollment Links

### Enroll Today (All FREE)

**NVIDIA Deep Learning Institute:**
1. Go to: https://www.nvidia.com/en-us/training/
2. Create account (free)
3. Search: "CUDA"
4. Enroll in: "Fundamentals of Accelerated Computing with CUDA C/C++"

**Coursera (Audit Mode):**
1. Search: "Heterogeneous Parallel Programming"
2. Click: "Enroll for Free"
3. Select: "Audit this course"
4. Access all materials!

**YouTube Playlists:**
1. Search: "CUDA tutorial playlist"
2. Save to "Watch Later"
3. Watch at 1.5x speed

---

## 📱 Mobile Learning Setup

### YouTube Channels to Subscribe

**Subscribe Now:**
1. NVIDIA Developer
2. CoffeeBeforeArch
3. Computerphile (GPU episodes)
4. TwoMinutePapers (GPU acceleration topics)

**Create Playlist:** "CUDA Learning"

**Download for offline:** When on WiFi

---

## 🔗 One-Click Access

### Copy These to Your Terminal Config

Add to `~/.zshrc`:

```bash
# CUDA Quick Links
alias cuda-docs='xdg-open https://docs.nvidia.com/cuda/'
alias cuda-forum='xdg-open https://forums.developer.nvidia.com/'
alias cuda-samples='cd ~/cuda-learning/external-resources/cuda-samples'
alias cuda-learn='cd ~/cuda-learning/01-basics'
alias cuda-progress='nvim ~/cuda-learning/PROGRESS.md'
alias cuda-resources='cat ~/cuda-learning/resources/FREE-RESOURCES.md | less'
```

Then:
```bash
source ~/.zshrc
cuda-docs    # Opens browser to docs
cuda-learn   # Goes to your lessons
```

---

## 🎯 Daily Learning Routine with Links

### Morning (30 min)
1. Open: PROGRESS.md (check today's lesson)
2. Open: Lesson file in editor
3. Optional: Read relevant Programming Guide chapter

### Afternoon (60 min)
1. Code the lesson
2. Stuck? → Stack Overflow search
3. Confused? → Programming Guide section
4. Need example? → CUDA samples

### Evening (30 min)
1. Do exercises
2. Check: GTC video on topic (optional)
3. Update: PROGRESS.md
4. Preview: Tomorrow's lesson

---

## 📊 Milestone Resources

### After Week 1
- Start: NVIDIA DLI course
- Read: Best Practices intro
- Watch: "Introduction to CUDA" GTC talks

### After Week 2
- Read: "Optimizing Memory" blog posts
- Profile: Your code with NSight
- Study: CUDA samples (memory)

### After Week 4
- Watch: Advanced GTC sessions
- Read: Research papers
- Implement: Paper algorithms

### After Week 8
- Contribute: GitHub project
- Answer: Stack Overflow questions
- Share: Your learning blog

---

## 🌟 Community Engagement

### Give Back (When Ready)

**Stack Overflow:**
- Search: "cuda [newest]"
- Answer: Questions you know
- Learn: By teaching

**GitHub:**
- Star: Useful repos
- Fork: Make improvements
- Share: Your projects

**Reddit:**
- Post: Your progress
- Help: Other learners
- Discuss: Techniques

---

```
╔══════════════════════════════════════════════╗
║  Bookmark This File                          ║
║  Use It Every Day                            ║
║  Update As You Find New Resources            ║
╚══════════════════════════════════════════════╝
```

**Your toolkit is complete. All resources at your fingertips.**
