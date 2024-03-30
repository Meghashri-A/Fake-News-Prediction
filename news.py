import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, redirect, url_for

# Load the dataset
df = pd.read_csv('news.csv')  # Replace 'news.csv' with your dataset file

# Assuming your dataset has 'title' as the second column containing titles and 'text' as the third column containing news articles, and 'label' for classification
X = df[['title', 'text']]  # Combine 'title' and 'text' columns
y = df['label']

# Combine title and text columns for both train and test sets
X_combined = X['title'] + ' ' + X['text']

# Initialize a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the data
tfidf_data = tfidf_vectorizer.fit_transform(X_combined)

# Initialize Logistic Regression Classifier
lr = LogisticRegression(max_iter=1000)

# Fit the model
lr.fit(tfidf_data, y)

news= Flask(__name__)

@news.route('/')
def index():
    return render_template('index.html')

@news.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Your authentication logic here
        # Placeholder logic for demonstration
        if username == 'user' and password == 'password':
            return redirect(url_for('interface'))
        else:
            return render_template('login.html', message='Invalid credentials. Please try again.')
    return render_template('login.html')

@news.route('/interface', methods=['GET', 'POST'])
def interface():
    predicted_label = None
    if request.method == 'POST':
        news_title = request.form['title']
        news_text = request.form['text']
        combined_news = news_title + ' ' + news_text
        tfidf_combined = tfidf_vectorizer.transform([combined_news])
        predicted_label = lr.predict(tfidf_combined)
        print(f'Predicted Label: {predicted_label}')
    return render_template('interface.html', result=predicted_label)


if __name__ == '__main__':
    news.run(debug=True)

    
'''    
# Classify new data using Logistic Regression
news_title = "A man has died and two others, including a British man, have been injured in a knife and hammer attack on a street in central Paris."
news_text = """
France's Interior Minister Gérald Darmanin said the victim was with his wife when he was attacked and fatally stabbed on Quai de Grenelle.
He said the wife's life was saved by the intervention of a taxi driver and that the suspect fled across a nearby bridge spanning the River Seine.
After crossing to the north side of the river he attacked two more people, hitting the 66-year-old British victim in the eye with a hammer.
The suspect was then Tasered by police and arrested on suspicion of assassination - defined in French law as premeditated murder - and "attempted assassination in relation to a terrorist enterprise".
Video published online appeared to show the moment the suspect was apprehended by armed police not far from where the attack happened.
He has been named in French media as Armand R, a 26-year-old French national with Iranian parents.
The two people injured - a Frenchman aged around 60 and a British tourist - were treated by emergency services, with neither found to be in a life-threatening condition.
On Sunday, Health Minister Aurélien Rousseau told French media the pair are "in good health".
A police operation was initiated around the Bir-Hakeim metro station on Saturday night, and authorities urged people to avoid the area.
Mr Darmanin said the alleged attacker was heard shouting "Allahu Akbar", Arabic for "God is greatest", and told police he was upset because "so many Muslims are dying in Afghanistan and in Palestine".
The suspect is also understood to have suggested France was complicit in the deaths of Palestinians in Gaza.
Police said the man was released from prison in 2020 after serving four years for planning an attack, and was supposed to be following treatment for psychiatric problems.
On Saturday, a video was posted on social media in which the suspect criticised the French government and discussed what he described as the murder of innocent Muslims, AFP news agency reports.
Writing on X, formerly Twitter, French President Emmanuel Macron sent his thoughts to all those affected by the "terrorist attack" and thanked the emergency services for their response.
"The national anti-terrorist prosecutor's office will now be responsible for shedding light on this affair so that justice can be done in the name of the French people," he said.
It comes less than two months after a teacher was killed in a knife attack at a high school in the northern city of Arras, prompting the French government to put the country on its highest level of national security alert.
"""

combined_news = news_title + ' ' + news_text
tfidf_combined = tfidf_vectorizer.transform([combined_news])

predicted_label = lr.predict(tfidf_combined)
print(f'Predicted Label: {predicted_label}')'''
