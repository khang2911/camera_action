#!/bin/bash

# Script to push code to GitHub with branch support
# Usage: 
#   ./push_to_github.sh [commit_message]              # Push current branch
#   ./push_to_github.sh -b <branch_name> [commit_msg] # Push specific branch
#   ./push_to_github.sh --all                          # Push all branches
#   ./push_to_github.sh --list                         # List all branches

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

# Parse arguments
PUSH_ALL=false
LIST_BRANCHES=false
TARGET_BRANCH=""
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            PUSH_ALL=true
            shift
            ;;
        --list)
            LIST_BRANCHES=true
            shift
            ;;
        -b|--branch)
            TARGET_BRANCH="$2"
            shift 2
            ;;
        *)
            if [ -z "$COMMIT_MSG" ]; then
                COMMIT_MSG="$1"
            fi
            shift
            ;;
    esac
done

# Handle list branches
if [ "$LIST_BRANCHES" = true ]; then
    echo -e "${GREEN}=== Local Branches ===${NC}"
    git branch
    echo ""
    echo -e "${GREEN}=== Remote Branches ===${NC}"
    git branch -r 2>/dev/null || echo "No remote branches found"
    echo ""
    echo -e "${GREEN}=== All Branches ===${NC}"
    git branch -a 2>/dev/null || echo "No branches found"
    exit 0
fi

# Get current branch if not specified
if [ -z "$TARGET_BRANCH" ]; then
    TARGET_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
fi

# Validate branch exists
if ! git show-ref --verify --quiet refs/heads/"$TARGET_BRANCH" && [ "$PUSH_ALL" = false ]; then
    echo -e "${RED}Error: Branch '$TARGET_BRANCH' does not exist locally.${NC}"
    echo "Available branches:"
    git branch
    exit 1
fi

# Handle push all branches
if [ "$PUSH_ALL" = true ]; then
    echo -e "${GREEN}=== Pushing All Branches ===${NC}"
    echo ""
    branches=$(git branch --format='%(refname:short)')
    for branch in $branches; do
        echo -e "${YELLOW}Pushing branch: $branch${NC}"
        git checkout "$branch" 2>/dev/null || continue
        if [ -n "$(git status --porcelain)" ]; then
            echo -e "${YELLOW}  Branch has uncommitted changes, skipping...${NC}"
            continue
        fi
        if git push -u origin "$branch" 2>/dev/null; then
            echo -e "${GREEN}  ✓ Pushed: $branch${NC}"
        else
            echo -e "${RED}  ✗ Failed: $branch${NC}"
        fi
        echo ""
    done
    # Return to original branch
    git checkout "$TARGET_BRANCH" 2>/dev/null || true
    echo -e "${GREEN}✓ Finished pushing all branches${NC}"
    exit 0
fi

# Switch to target branch if different from current
current_branch=$(git branch --show-current 2>/dev/null || echo "main")
if [ "$current_branch" != "$TARGET_BRANCH" ]; then
    echo -e "${YELLOW}Switching to branch: $TARGET_BRANCH${NC}"
    git checkout "$TARGET_BRANCH" || {
        echo -e "${RED}Error: Failed to checkout branch '$TARGET_BRANCH'${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Switched to branch: $TARGET_BRANCH${NC}"
    echo ""
fi

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
    
    # Stage all changes except ONNX models
    echo "Staging all changes (excluding *.onnx)..."
    git add .
    # Unstage any ONNX files to avoid pushing heavy models
    if git ls-files -o -m --exclude-standard | grep -Ei '\.onnx$' >/dev/null 2>&1 || \
       git diff --name-only --cached | grep -Ei '\.onnx$' >/dev/null 2>&1; then
        git reset HEAD -- '*.onnx' >/dev/null 2>&1 || true
        echo -e "${YELLOW}Note: *.onnx files were detected and left unstaged to avoid pushing.${NC}"
    fi
    
    # Get commit message
    if [ -n "$COMMIT_MSG" ]; then
        commit_msg="$COMMIT_MSG"
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
if git ls-remote --heads origin "$TARGET_BRANCH" 2>/dev/null | grep -q "$TARGET_BRANCH"; then
    echo -e "${YELLOW}Branch '$TARGET_BRANCH' exists on remote.${NC}"
    read -p "Pull latest changes first? (y/n): " pull_first
    if [ "$pull_first" = "y" ]; then
        echo "Pulling latest changes..."
        git pull origin "$TARGET_BRANCH" --no-rebase || {
            echo -e "${RED}Warning: Pull failed. You may have conflicts.${NC}"
            read -p "Continue with push anyway? (y/n): " continue_push
            if [ "$continue_push" != "y" ]; then
                exit 1
            fi
        }
    fi
else
    echo -e "${GREEN}Branch '$TARGET_BRANCH' is new on remote.${NC}"
    read -p "Create and push this new branch? (y/n): " create_branch
    if [ "$create_branch" != "y" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo "Repository: $(git remote get-url origin)"
echo "Branch: $TARGET_BRANCH"
echo ""

if git push -u origin "$TARGET_BRANCH"; then
    echo ""
    echo -e "${GREEN}✓ Successfully pushed to GitHub!${NC}"
    echo ""
    echo "Repository: $(git remote get-url origin)"
    echo "Branch: $TARGET_BRANCH"
    echo ""
    echo "View on GitHub:"
    repo_url=$(git remote get-url origin)
    # Convert SSH to HTTPS URL for display
    if [[ "$repo_url" == git@github.com:* ]]; then
        repo_url=$(echo "$repo_url" | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/\.git$//')
    fi
    echo "  $repo_url/tree/$TARGET_BRANCH"
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

