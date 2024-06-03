from glob import glob
import shutil
import os
import git
import requests

languages = ['Python', 'C++', 'Javascript']
extensions = {
    'Python': ['*.py'],
    'C++': ['*.cpp', '*.h' '*.hpp', '*.cxx'],
    'Javascript': ['*.js']
}

repos_per_langauge = 5


def fetch_repositories(language, num_results):
    url = f"https://api.github.com/search/repositories?q=language:{language}&sort=stars&order=desc"
    repos = requests.get(url, params={'per_page': num_results})
    data = repos.json()
    return data['items']


def pull(language, url, masks):
    repo_name = url.split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]

    root_dir = os.path.join(os.getcwd(), f'sources\\{language}')

    repo_path = os.path.join(root_dir, repo_name)
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    print('\tCloning...')
    git.Repo.clone_from(url=url, to_path=repo_path)

    dest_path = os.path.join(root_dir, f"{language}")
    os.makedirs(dest_path, exist_ok=True)
    print('\tFiltering...')
    for mask in masks:
        for file_path in glob(os.path.join(repo_path, '**', mask), recursive=True):
            shutil.copy(file_path, dest_path)

    shutil.rmtree(repo_path)

    with open(os.path.join(root_dir, f'{language}.txt'), 'a') as f:
        f.write(f'{url}\n')


if __name__ == '__main__':
    for lang in languages:
        repos = fetch_repositories(lang, repos_per_langauge)
        for repo in repos:
            full_name = repo['full_name']
            print(f'{lang}:\t{full_name}')
            pull(lang, repo['git_url'], extensions[lang])
