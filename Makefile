.PHONY: push-www help

help:
	@echo "VHF Tools Deployment"
	@echo ""
	@echo "  make push-www    Deploy to www server via git pull"
	@echo ""

push-www:
	@echo "Deploying VHF tools to www server via git pull..."
	@echo ""
	@echo "Step 1: Pushing to GitHub..."
	@git push origin main
	@echo ""
	@echo "Step 2: Pulling on www server..."
	@ssh hftools@www 'cd ~/repos/ak6mj-vhf-tools && git pull origin main'
	@echo ""
	@echo "Step 3: Restarting service..."
	@ssh root@www 'systemctl restart vhf-web && systemctl status vhf-web --no-pager | head -10'
	@echo ""
	@echo "✓ Deployment complete!"
	@echo ""
	@echo "Note: Templates and vhf_webapp.py are symlinked from ~/repos/ak6mj-vhf-tools/"
	@echo "Access at: https://www.shoeph.one/vhf/"
