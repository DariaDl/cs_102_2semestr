from sqlalchemy import Column, Integer, String, create_engine  # type: ignore
from sqlalchemy.orm import sessionmaker  # type: ignore
from sqlalchemy.ext.declarative import declarative_base  # type: ignore

from scraputils import get_news # type: ignore
Base = declarative_base()  # type: ignore
engine = create_engine("sqlite:///news.db")  # type: ignore
session = sessionmaker(bind=engine)  # type: ignore


class News(Base):  # type: ignore
    __tablename__ = "news"  # type: ignore
    id = Column(Integer, primary_key=True)
    title = Column(String)
    author = Column(String)
    url = Column(String)
    comments = Column(Integer)
    points = Column(Integer)
    label = Column(String)


Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    k = session()
    news_list = get_news("https://news.ycombinator.com/newest", n_pages=35)
    for element in range(len(news_list)):
        news = News(
            title=news_list[element]["title"],
            author=news_list[element]["author"],
            url=news_list[element]["url"],
            comments=news_list[element]["comments"],
            points=news_list[element]["points"],
        )
        k.add(news)
        k.commit()

