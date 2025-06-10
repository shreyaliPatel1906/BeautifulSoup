from bs4 import BeautifulSoup
def load_html(file_path):
    with open(file_path, 'r',encoding="utf-8") as f:
        content = f.read()
        return content

def extract_posts(soup):
    posts = []
    post_elements = soup.find_all(name='div', class_='person')
    # print(post_elements)
    for post in post_elements:
        name = post.find(name='h1').text
        city = post.find(name='h2').text
        company = post.find(name='h3').text
        designation = post.find(name='p').text

        # print(username, content ,timestamp )

        posts.append({'name': name, 'city': city, 'company': company, 'designation': designation})
    return posts

html_content = load_html("person.html")
soup = BeautifulSoup(html_content,features="html.parser")

posts = extract_posts(soup)
# print(posts)

for post in posts:
    print(f"name: {post['name']}")
    print(f"city: {post['city']}")
    print(f"company: {post['company']}")
    print(f"designation: {post['designation']}")