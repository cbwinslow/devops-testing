#!/usr/bin/env python3

import tarfile
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path
import logging
import asyncio
from typing import Dict, List, Optional
import os
import hashlib
import base64
from cryptography.fernet import Fernet
import click

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_backup')

class BackupManager:
    """Manage system backups and restoration"""
    def __init__(self, config_path: str = 'config/quantum_config.yaml'):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.setup_backup_dir()
        self.setup_encryption()

    def load_config(self) -> Dict:
        """Load backup configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def setup_backup_dir(self):
        """Setup backup directory"""
        self.backup_dir = Path('backups')
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def setup_encryption(self):
        """Setup encryption for sensitive data"""
        try:
            key_file = Path('.backup_key')
            if not key_file.exists():
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
            else:
                with open(key_file, 'rb') as f:
                    key = f.read()
            self.cipher = Fernet(key)
        except Exception as e:
            logger.error(f"Error setting up encryption: {e}")
            self.cipher = None

    def create_backup(self, backup_type: str = 'full') -> str:
        """Create system backup"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"quantum_backup_{backup_type}_{timestamp}"
            backup_path = self.backup_dir / backup_name

            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)

            # Paths to backup
            paths_to_backup = {
                'config': Path('config'),
                'data': Path('data'),
                'logs': Path('logs'),
                'reports': Path('reports'),
                'visualizations': Path('visualizations')
            }

            # Backup metadata
            metadata = {
                'timestamp': timestamp,
                'type': backup_type,
                'paths': list(paths_to_backup.keys()),
                'checksum': {}
            }

            # Create archives for each component
            for name, path in paths_to_backup.items():
                if path.exists():
                    archive_path = backup_path / f"{name}.tar.gz"
                    with tarfile.open(archive_path, 'w:gz') as tar:
                        tar.add(path, arcname=name)
                    
                    # Calculate checksum
                    with open(archive_path, 'rb') as f:
                        checksum = hashlib.sha256(f.read()).hexdigest()
                        metadata['checksum'][name] = checksum

            # Encrypt sensitive data
            if self.cipher:
                sensitive_paths = ['config/quantum_config.yaml']
                for path in sensitive_paths:
                    if Path(path).exists():
                        with open(path, 'rb') as f:
                            data = f.read()
                        encrypted_data = self.cipher.encrypt(data)
                        encrypted_path = backup_path / f"{Path(path).name}.encrypted"
                        with open(encrypted_path, 'wb') as f:
                            f.write(encrypted_data)

            # Save metadata
            metadata_path = backup_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Create final archive
            final_archive = self.backup_dir / f"{backup_name}.tar.gz"
            with tarfile.open(final_archive, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_name)

            # Cleanup temporary directory
            shutil.rmtree(backup_path)

            logger.info(f"Backup created: {final_archive}")
            return str(final_archive)

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return ""

    def restore_backup(self, backup_path: str, restore_type: str = 'full') -> bool:
        """Restore system from backup"""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_path}")
                return False

            # Create temporary directory for restoration
            temp_dir = Path('temp_restore')
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Extract backup archive
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(temp_dir)

            # Load metadata
            backup_name = backup_path.stem
            metadata_path = temp_dir / backup_name / 'metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Verify checksums
            for name, checksum in metadata['checksum'].items():
                archive_path = temp_dir / backup_name / f"{name}.tar.gz"
                if archive_path.exists():
                    with open(archive_path, 'rb') as f:
                        current_checksum = hashlib.sha256(f.read()).hexdigest()
                        if current_checksum != checksum:
                            logger.error(f"Checksum mismatch for {name}")
                            return False

            # Restore components
            for name in metadata['paths']:
                archive_path = temp_dir / backup_name / f"{name}.tar.gz"
                if archive_path.exists():
                    # Create target directory
                    target_dir = Path(name)
                    target_dir.mkdir(parents=True, exist_ok=True)

                    # Extract component
                    with tarfile.open(archive_path, 'r:gz') as tar:
                        tar.extractall('.')

            # Restore encrypted files
            if self.cipher:
                encrypted_files = list((temp_dir / backup_name).glob('*.encrypted'))
                for encrypted_file in encrypted_files:
                    with open(encrypted_file, 'rb') as f:
                        encrypted_data = f.read()
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    target_path = Path('config') / encrypted_file.stem
                    with open(target_path, 'wb') as f:
                        f.write(decrypted_data)

            # Cleanup
            shutil.rmtree(temp_dir)

            logger.info(f"Backup restored successfully from: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False

    def list_backups(self) -> List[Dict]:
        """List available backups"""
        try:
            backups = []
            for backup_file in self.backup_dir.glob('quantum_backup_*.tar.gz'):
                try:
                    # Extract metadata
                    with tarfile.open(backup_file, 'r:gz') as tar:
                        metadata_member = next(
                            (m for m in tar.getmembers() if m.name.endswith('metadata.json')),
                            None
                        )
                        if metadata_member:
                            metadata = json.load(tar.extractfile(metadata_member))
                            backups.append({
                                'file': str(backup_file),
                                'timestamp': metadata['timestamp'],
                                'type': metadata['type'],
                                'size': backup_file.stat().st_size,
                                'paths': metadata['paths']
                            })
                except Exception as e:
                    logger.warning(f"Error reading backup {backup_file}: {e}")

            return sorted(backups, key=lambda x: x['timestamp'], reverse=True)

        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []

    def verify_backup(self, backup_path: str) -> Dict:
        """Verify backup integrity"""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                return {'valid': False, 'error': 'Backup file not found'}

            # Create temporary directory for verification
            temp_dir = Path('temp_verify')
            temp_dir.mkdir(parents=True, exist_ok=True)

            verification_results = {
                'valid': True,
                'components': {},
                'errors': []
            }

            try:
                # Extract backup archive
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.extractall(temp_dir)

                # Load metadata
                backup_name = backup_path.stem
                metadata_path = temp_dir / backup_name / 'metadata.json'
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Verify each component
                for name, checksum in metadata['checksum'].items():
                    archive_path = temp_dir / backup_name / f"{name}.tar.gz"
                    if archive_path.exists():
                        with open(archive_path, 'rb') as f:
                            current_checksum = hashlib.sha256(f.read()).hexdigest()
                            is_valid = current_checksum == checksum
                            verification_results['components'][name] = {
                                'valid': is_valid,
                                'expected_checksum': checksum,
                                'actual_checksum': current_checksum
                            }
                            if not is_valid:
                                verification_results['valid'] = False
                                verification_results['errors'].append(
                                    f"Checksum mismatch for {name}"
                                )
                    else:
                        verification_results['components'][name] = {
                            'valid': False,
                            'error': 'Component archive not found'
                        }
                        verification_results['valid'] = False
                        verification_results['errors'].append(
                            f"Missing component archive: {name}"
                        )

            finally:
                # Cleanup
                shutil.rmtree(temp_dir)

            return verification_results

        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return {
                'valid': False,
                'error': str(e)
            }

    def cleanup_old_backups(self, days: int) -> int:
        """Remove backups older than specified days"""
        try:
            removed_count = 0
            cutoff_time = datetime.now().timestamp() - (days * 86400)
            
            for backup_file in self.backup_dir.glob('quantum_backup_*.tar.gz'):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    removed_count += 1
                    logger.info(f"Removed old backup: {backup_file}")
            
            return removed_count

        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            return 0

# CLI Commands
def add_backup_commands(cli):
    @cli.group()
    def backup():
        """Backup and restore commands"""
        pass

    @backup.command('create')
    @click.option('--type', type=click.Choice(['full', 'config', 'data']), default='full')
    def backup_create(type):
        """Create system backup"""
        manager = BackupManager()
        backup_path = manager.create_backup(type)
        if backup_path:
            click.echo(f"Backup created successfully: {backup_path}")
        else:
            click.echo("Failed to create backup")

    @backup.command('restore')
    @click.argument('backup-path')
    @click.option('--type', type=click.Choice(['full', 'config', 'data']), default='full')
    def backup_restore(backup_path, type):
        """Restore system from backup"""
        if click.confirm('This will overwrite existing data. Continue?'):
            manager = BackupManager()
            success = manager.restore_backup(backup_path, type)
            if success:
                click.echo("System restored successfully")
            else:
                click.echo("Failed to restore system")

    @backup.command('list')
    def backup_list():
        """List available backups"""
        manager = BackupManager()
        backups = manager.list_backups()
        if backups:
            click.echo("\nAvailable Backups:")
            for backup in backups:
                click.echo(f"\nFile: {backup['file']}")
                click.echo(f"Type: {backup['type']}")
                click.echo(f"Timestamp: {backup['timestamp']}")
                click.echo(f"Size: {backup['size'] / 1024 / 1024:.2f} MB")
                click.echo(f"Components: {', '.join(backup['paths'])}")
        else:
            click.echo("No backups found")

    @backup.command('verify')
    @click.argument('backup-path')
    def backup_verify(backup_path):
        """Verify backup integrity"""
        manager = BackupManager()
        results = manager.verify_backup(backup_path)
        if results['valid']:
            click.echo("\nBackup verification successful!")
            click.echo("\nComponent Status:")
            for name, status in results['components'].items():
                if status['valid']:
                    click.echo(f"- {name}: {click.style('OK', fg='green')}")
                else:
                    click.echo(f"- {name}: {click.style('FAILED', fg='red')}")
                    if 'error' in status:
                        click.echo(f"  Error: {status['error']}")
        else:
            click.echo("\nBackup verification failed!")
            for error in results.get('errors', []):
                click.echo(f"- {error}")

    @backup.command('cleanup')
    @click.option('--days', type=int, default=30, help='Remove backups older than N days')
    def backup_cleanup(days):
        """Clean up old backups"""
        manager = BackupManager()
        removed = manager.cleanup_old_backups(days)
        click.echo(f"Removed {removed} old backup(s)")

if __name__ == '__main__':
    # This allows testing the backup functionality independently
    manager = BackupManager()
    backup_path = manager.create_backup()
    if backup_path:
        print(f"Test backup created: {backup_path}")
        verified = manager.verify_backup(backup_path)
        print("Verification results:", json.dumps(verified, indent=2))

