#!/bin/bash

# Simple script to push code to GitHub
# Usage: ./push_to_github.sh [commit_message]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Push to GitHub ===${NC}"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not a git repository${NC}"
    echo "Run 'git init' first or use setup_github.sh"
    exit 1
fi

# Check if remote exists
if ! git remote | grep -q "^origin$"; then
    echo -e "${YELLOW}No remote 'origin' found.${NC}"
    read -p "Enter GitHub repository URL: " repo_url
    if [ -z "$repo_url" ]; then
        echo -e "${RED}No URL provided. Exiting.${NC}"
        exit 1
    fi
    git remote add origin "$repo_url"
    echo -e "${GREEN}✓ Remote 'origin' added${NC}"
fi

# Get current branch
current_branch=$(git branch --show-current 2>/dev/null || echo "main")

# Check for changes
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}No changes to commit.${NC}"
    read -p "Do you want to push anyway? (y/n): " push_anyway
    if [ "$push_anyway" != "y" ]; then
        exit 0
    fi
else
    # Show status
    echo "Changes to be committed:"
    git status --short
    echo ""
    
    # Stage all changes
    echo "Staging all changes..."
    git add .
    
    # Get commit message
    if [ -n "$1" ]; then
        commit_msg="$1"
    else
        read -p "Enter commit message: " commit_msg
        if [ -z "$commit_msg" ]; then
            commit_msg="Update: $(date '+%Y-%m-%d %H:%M:%S')"
        fi
    fi
    
    # Commit
    echo "Committing changes..."
    git commit -m "$commit_msg"
    echo -e "${GREEN}✓ Committed: $commit_msg${NC}"
fi

# Check if branch exists on remote
echo ""
echo "Checking remote status..."
if git ls-remote --heads origin "$current_branch" 2>/dev/null | grep -q "$current_branch"; then
    echo -e "${YELLOW}Branch '$current_branch' exists on remote.${NC}"
    read -p "Pull latest changes first? (y/n): " pull_first
    if [ "$pull_first" = "y" ]; then
        echo "Pulling latest changes..."
        git pull origin "$current_branch" --no-rebase || {
            echo -e "${RED}Warning: Pull failed. You may have conflicts.${NC}"
            read -p "Continue with push anyway? (y/n): " continue_push
            if [ "$continue_push" != "y" ]; then
                exit 1
            fi
        }
    fi
else
    echo -e "${GREEN}Branch '$current_branch' is new on remote.${NC}"
fi

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo "Repository: $(git remote get-url origin)"
echo "Branch: $current_branch"
echo ""

if git push -u origin "$current_branch"; then
    echo ""
    echo -e "${GREEN}✓ Successfully pushed to GitHub!${NC}"
    echo ""
    echo "Repository: $(git remote get-url origin)"
    echo "Branch: $current_branch"
    echo ""
    echo "View on GitHub:"
    repo_url=$(git remote get-url origin)
    # Convert SSH to HTTPS URL for display
    if [[ "$repo_url" == git@github.com:* ]]; then
        repo_url=$(echo "$repo_url" | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/\.git$//')
    fi
    echo "  $repo_url"
else
    echo ""
    echo -e "${RED}✗ Push failed!${NC}"
    echo ""
    echo "Common issues:"
    echo "  1. Authentication failed - check your SSH keys or GitHub token"
    echo "  2. Repository doesn't exist - create it on GitHub first"
    echo "  3. Permission denied - check repository access"
    echo ""
    echo "Troubleshooting:"
    echo "  - Test SSH: ssh -T git@github.com"
    echo "  - Check remote: git remote -v"
    echo "  - View logs: git log --oneline -5"
    exit 1
fi

