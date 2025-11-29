# GitHub Setup Instructions

## Quick Setup (Using Script)

Run the setup script:
```bash
./setup_github.sh
```

Or with repository URL:
```bash
./setup_github.sh https://github.com/yourusername/your-repo-name.git
```

## Manual Setup

### Step 1: Create Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Enter repository name (e.g., "ai-camera-solution")
4. Choose public or private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Step 2: Add Remote and Push

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Or if using SSH:
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

### Step 3: Verify

Check your repository on GitHub - all files should be there!

## Authentication

### Option 1: Personal Access Token (HTTPS)
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use token as password when pushing

### Option 2: SSH Keys (Recommended)
1. Generate SSH key if you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
2. Add to SSH agent:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```
3. Add public key to GitHub: Settings → SSH and GPG keys → New SSH key
4. Use SSH URL for remote: `git@github.com:username/repo.git`

### Option 3: GitHub CLI
```bash
# Install GitHub CLI, then:
gh auth login
gh repo create ai-camera-solution --public --source=. --remote=origin --push
```

## Current Status

✅ Git repository initialized
✅ All files staged and committed
✅ Ready to push to GitHub

## Next Steps

1. Create repository on GitHub (if not done)
2. Run: `./setup_github.sh` or follow manual steps above
3. Push your code!

