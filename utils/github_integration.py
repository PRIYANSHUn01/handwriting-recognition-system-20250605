import os
import json
import base64
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
import streamlit as st

class GitHubIntegration:
    """GitHub integration for handwriting system version control and sharing"""
    
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        self.base_url = 'https://api.github.com'
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        
    def is_authenticated(self) -> bool:
        """Check if GitHub authentication is valid"""
        if not self.token:
            return False
        
        try:
            response = requests.get(f'{self.base_url}/user', headers=self.headers)
            return response.status_code == 200
        except:
            return False
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get authenticated user information"""
        try:
            response = requests.get(f'{self.base_url}/user', headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error getting user info: {str(e)}")
            return None
    
    def list_repositories(self, type='owner') -> List[Dict[str, Any]]:
        """List user repositories"""
        try:
            response = requests.get(
                f'{self.base_url}/user/repos?type={type}&sort=updated&per_page=100',
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.error(f"Error listing repositories: {str(e)}")
            return []
    
    def create_repository(self, name: str, description: str = "", private: bool = False) -> Optional[Dict[str, Any]]:
        """Create a new repository"""
        try:
            data = {
                'name': name,
                'description': description,
                'private': private,
                'auto_init': True,
                'gitignore_template': 'Python'
            }
            
            response = requests.post(
                f'{self.base_url}/user/repos',
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                st.error(f"Error creating repository: {response.json().get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error creating repository: {str(e)}")
            return None
    
    def upload_file(self, repo_owner: str, repo_name: str, file_path: str, 
                   content: str, commit_message: str, branch: str = 'main') -> bool:
        """Upload file to repository"""
        try:
            # Encode content to base64
            encoded_content = base64.b64encode(content.encode()).decode()
            
            # Check if file exists to get SHA
            file_url = f'{self.base_url}/repos/{repo_owner}/{repo_name}/contents/{file_path}'
            response = requests.get(file_url, headers=self.headers)
            
            data = {
                'message': commit_message,
                'content': encoded_content,
                'branch': branch
            }
            
            # If file exists, include SHA for update
            if response.status_code == 200:
                existing_file = response.json()
                data['sha'] = existing_file['sha']
            
            # Upload/update file
            response = requests.put(file_url, headers=self.headers, json=data)
            
            if response.status_code in [200, 201]:
                return True
            else:
                st.error(f"Error uploading file: {response.json().get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            return False
    
    def download_file(self, repo_owner: str, repo_name: str, file_path: str, branch: str = 'main') -> Optional[str]:
        """Download file content from repository"""
        try:
            file_url = f'{self.base_url}/repos/{repo_owner}/{repo_name}/contents/{file_path}?ref={branch}'
            response = requests.get(file_url, headers=self.headers)
            
            if response.status_code == 200:
                file_data = response.json()
                # Decode base64 content
                content = base64.b64decode(file_data['content']).decode()
                return content
            return None
            
        except Exception as e:
            st.error(f"Error downloading file: {str(e)}")
            return None
    
    def save_handwriting_style(self, repo_owner: str, repo_name: str, style_name: str, 
                              style_data: Dict[str, Any], session_id: str) -> bool:
        """Save custom handwriting style to GitHub repository"""
        try:
            # Prepare style data with metadata
            export_data = {
                'style_name': style_name,
                'style_data': style_data,
                'metadata': {
                    'created_by': session_id,
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'type': 'handwriting_style'
                }
            }
            
            # Convert to JSON
            content = json.dumps(export_data, indent=2)
            
            # Upload to repository
            file_path = f'handwriting_styles/{style_name}.json'
            commit_message = f"Add handwriting style: {style_name}"
            
            return self.upload_file(repo_owner, repo_name, file_path, content, commit_message)
            
        except Exception as e:
            st.error(f"Error saving handwriting style: {str(e)}")
            return False
    
    def load_handwriting_style(self, repo_owner: str, repo_name: str, style_name: str) -> Optional[Dict[str, Any]]:
        """Load custom handwriting style from GitHub repository"""
        try:
            file_path = f'handwriting_styles/{style_name}.json'
            content = self.download_file(repo_owner, repo_name, file_path)
            
            if content:
                return json.loads(content)
            return None
            
        except Exception as e:
            st.error(f"Error loading handwriting style: {str(e)}")
            return None
    
    def save_neural_network_model(self, repo_owner: str, repo_name: str, model_name: str, 
                                 model_data: Dict[str, Any], session_id: str) -> bool:
        """Save trained neural network model to GitHub repository"""
        try:
            # Prepare model data with metadata
            export_data = {
                'model_name': model_name,
                'model_data': model_data,
                'metadata': {
                    'created_by': session_id,
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'type': 'neural_network_model'
                }
            }
            
            # Convert to JSON
            content = json.dumps(export_data, indent=2)
            
            # Upload to repository
            file_path = f'neural_models/{model_name}.json'
            commit_message = f"Add neural network model: {model_name}"
            
            return self.upload_file(repo_owner, repo_name, file_path, content, commit_message)
            
        except Exception as e:
            st.error(f"Error saving neural network model: {str(e)}")
            return False
    
    def backup_user_data(self, repo_owner: str, repo_name: str, user_data: Dict[str, Any], 
                        session_id: str) -> bool:
        """Backup user handwriting data to GitHub repository"""
        try:
            # Prepare backup data
            backup_data = {
                'session_id': session_id,
                'backup_date': datetime.now().isoformat(),
                'user_data': user_data,
                'metadata': {
                    'type': 'user_backup',
                    'version': '1.0'
                }
            }
            
            # Convert to JSON
            content = json.dumps(backup_data, indent=2)
            
            # Upload to repository
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f'backups/user_data_{timestamp}.json'
            commit_message = f"Backup user data - {timestamp}"
            
            return self.upload_file(repo_owner, repo_name, file_path, content, commit_message)
            
        except Exception as e:
            st.error(f"Error backing up user data: {str(e)}")
            return False
    
    def list_shared_styles(self, repo_owner: str, repo_name: str) -> List[str]:
        """List available handwriting styles in repository"""
        try:
            # Get contents of handwriting_styles directory
            dir_url = f'{self.base_url}/repos/{repo_owner}/{repo_name}/contents/handwriting_styles'
            response = requests.get(dir_url, headers=self.headers)
            
            if response.status_code == 200:
                files = response.json()
                style_names = []
                
                for file in files:
                    if file['name'].endswith('.json'):
                        style_name = file['name'].replace('.json', '')
                        style_names.append(style_name)
                
                return style_names
            return []
            
        except Exception as e:
            st.error(f"Error listing shared styles: {str(e)}")
            return []
    
    def create_handwriting_repository(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Create a dedicated repository for handwriting system data"""
        description = "Handwriting Recognition & Generation System - Custom styles and neural models"
        
        repo = self.create_repository(repo_name, description, private=False)
        
        if repo:
            # Create initial directory structure
            readme_content = """# Handwriting System Repository

This repository contains:
- Custom handwriting styles (`handwriting_styles/`)
- Trained neural network models (`neural_models/`)
- User data backups (`backups/`)

Generated by Handwriting Recognition & Generation System.
"""
            
            # Upload README
            self.upload_file(
                repo['owner']['login'], 
                repo['name'], 
                'README.md', 
                readme_content, 
                'Initial repository setup'
            )
            
            # Create directory structure with placeholder files
            placeholder_content = "# Placeholder file to maintain directory structure"
            
            directories = [
                'handwriting_styles/.gitkeep',
                'neural_models/.gitkeep',
                'backups/.gitkeep'
            ]
            
            for dir_file in directories:
                self.upload_file(
                    repo['owner']['login'], 
                    repo['name'], 
                    dir_file, 
                    placeholder_content, 
                    f'Create {dir_file.split("/")[0]} directory'
                )
        
        return repo
    
    def get_repository_stats(self, repo_owner: str, repo_name: str) -> Optional[Dict[str, Any]]:
        """Get repository statistics"""
        try:
            repo_url = f'{self.base_url}/repos/{repo_owner}/{repo_name}'
            response = requests.get(repo_url, headers=self.headers)
            
            if response.status_code == 200:
                repo_data = response.json()
                return {
                    'name': repo_data['name'],
                    'description': repo_data['description'],
                    'size': repo_data['size'],
                    'updated_at': repo_data['updated_at'],
                    'language': repo_data['language'],
                    'stars': repo_data['stargazers_count'],
                    'forks': repo_data['forks_count'],
                    'private': repo_data['private'],
                    'html_url': repo_data['html_url']
                }
            return None
            
        except Exception as e:
            st.error(f"Error getting repository stats: {str(e)}")
            return None