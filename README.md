# DevOps Testing and Validation System

A comprehensive system for automated testing, validation, monitoring, and error resolution in DevOps environments.

## Features

### 1. System Validation
- Directory structure validation
- Symlink verification
- Permission checks
- Package management tests
- Security mechanism validation

### 2. CI/CD Pipeline
- GitHub Actions workflow
- Automated testing
- Security scanning
- Documentation validation

### 3. Monitoring System
- Permission changes
- Configuration modifications
- Security alerts
- Package updates

### 4. AI-Powered Error Resolution
- Automated error detection
- Intelligent error analysis
- Automatic resolution attempts
- Detailed reporting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cbwinslow/devops-testing.git
cd devops-testing
```

2. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

3. Configure the system:
```bash
cp config/ai_resolver_config.yaml.example config/ai_resolver_config.yaml
# Edit config/ai_resolver_config.yaml with your settings
```

## Usage

### Running System Validation

```bash
bash tests/system_validation.sh
```

### Starting the Monitoring System

```bash
bash scripts/monitor.sh
```

### Running Tests

```bash
python3 -m pytest tests/
```

### Using the AI Resolver

The AI resolver automatically processes alerts from the monitoring system. To manually process alerts:

```bash
python3 scripts/ai_resolver.py /path/to/alerts.json
```

## Configuration

### AI Resolver Configuration

The AI resolver is configured through `config/ai_resolver_config.yaml`:

```yaml
notification:
  slack_webhook: "your-webhook-url"
  email: "your-email@example.com"

priorities:
  security: 5
  permission: 4
  config_change: 3
  package: 2
```

### Monitoring Configuration

Monitoring settings can be adjusted in `config/monitor_config.yaml`:

```yaml
monitoring:
  check_interval: 60  # seconds
  log_retention: 30   # days
  max_log_size: 1000  # MB
```

## Alert Types

1. **Security Alerts**
   - Failed login attempts
   - File integrity violations
   - Suspicious activity

2. **Permission Issues**
   - Incorrect file permissions
   - Directory access problems
   - Ownership mismatches

3. **Configuration Changes**
   - File modifications
   - Setting updates
   - Environment changes

4. **Package Management**
   - Missing packages
   - Update requirements
   - Dependency issues

## Resolution Process

1. **Alert Detection**
   - Continuous monitoring
   - Event logging
   - Priority assessment

2. **Analysis**
   - Context gathering
   - Pattern matching
   - Risk assessment

3. **Resolution**
   - Automated fixes
   - Rollback capabilities
   - Manual intervention triggers

4. **Reporting**
   - Detailed logs
   - Action summaries
   - Status updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Testing

The system includes comprehensive tests:

```bash
# Run all tests
python3 -m pytest

# Run with coverage
python3 -m pytest --cov=scripts tests/

# Run specific test file
python3 -m pytest tests/test_ai_resolver.py
```

## Logging

Logs are stored in the following locations:

- System validation: `reports/validation_*.log`
- Monitoring: `reports/monitoring/`
- AI resolver: `reports/ai_resolver.log`
- Resolutions: `reports/resolutions/`

## Security

The system implements several security measures:

1. Least privilege execution
2. Secure file handling
3. Configuration validation
4. Audit logging
5. Rollback capabilities

## Support

For issues and feature requests, please use the GitHub issue tracker.

## License

MIT License - see LICENSE file for details

