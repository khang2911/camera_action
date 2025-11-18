#!/bin/bash

# Script to help push code to GitHub
# Usage: ./setup_github.sh [github_repo_url]

echo "=== GitHub Setup Script ==="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "Staging all files..."
    git add .
    
    echo "Creating initial commit..."
    read -p "Enter commit message (or press Enter for default): " commit_msg
    if [ -z "$commit_msg" ]; then
        commit_msg="Initial commit: AI Camera Solution with multi-engine YOLO support"
    fi
    git commit -m "$commit_msg"
else
    echo "No changes to commit."
fi

# Check if remote exists
if git remote | grep -q "^origin$"; then
    echo ""
    echo "Remote 'origin' already exists:"
    git remote -v
    echo ""
    read -p "Do you want to update the remote URL? (y/n): " update_remote
    if [ "$update_remote" = "y" ]; then
        read -p "Enter new GitHub repository URL: " repo_url
        git remote set-url origin "$repo_url"
    fi
else
    if [ -n "$1" ]; then
        repo_url="$1"
    else
        echo ""
        echo "No remote repository configured."
        read -p "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): " repo_url
    fi
    
    if [ -n "$repo_url" ]; then
        echo "Adding remote repository..."
        git remote add origin "$repo_url"
    else
        echo "No repository URL provided. Skipping remote setup."
        echo "You can add it later with: git remote add origin <url>"
        exit 0
    fi
fi

# Get current branch name
current_branch=$(git branch --show-current 2>/dev/null || echo "main")

echo ""
echo "=== Ready to Push ==="
echo "Repository: $(git remote get-url origin 2>/dev/null || echo 'Not set')"
echo "Branch: $current_branch"
echo ""
read -p "Do you want to push to GitHub now? (y/n): " push_now

if [ "$push_now" = "y" ]; then
    # Check if branch exists on remote
    if git ls-remote --heads origin "$current_branch" | grep -q "$current_branch"; then
        echo "Branch exists on remote. Pulling latest changes..."
        git pull origin "$current_branch" --no-rebase || echo "Warning: Could not pull. Proceeding with push..."
    fi
    
    echo "Pushing to GitHub..."
    git push -u origin "$current_branch"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully pushed to GitHub!"
        echo "Repository URL: $(git remote get-url origin)"
    else
        echo ""
        echo "✗ Push failed. You may need to:"
        echo "  1. Create the repository on GitHub first"
        echo "  2. Check your authentication (SSH keys or GitHub CLI)"
        echo "  3. Try: git push -u origin $current_branch --force (if you're sure)"
    fi
else
    echo ""
    echo "To push manually, run:"
    echo "  git push -u origin $current_branch"
fi

echo ""
echo "=== Setup Complete ==="

