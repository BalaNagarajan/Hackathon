from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileWriterTool


load_dotenv()

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class AiSiteSummary():
	"""AiSiteSummary crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# Go head and do the google search of the news:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def retreive_news(self) -> Agent:
		return Agent(
			config=self.agents_config['retreive_news'],
			tools=[SerperDevTool()],
			verbose=True
		)

    # Go head and do the website scraping:
	@agent
	def website_scraper(self) -> Agent:
		return Agent(
			config=self.agents_config['website_scraper'],
			tools=[ScrapeWebsiteTool()],
			verbose=True
		)
	
	 # Go head and do the news writing:
	@agent
	def news_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['news_writer'],
			tools=[],
			verbose=True
		)
	

	 # Go head and do the website scraping:
	@agent
	def file_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['file_writer'],
			tools=[FileWriterTool()],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def retreive_news_task(self) -> Task:
		return Task(
			config=self.tasks_config['retreive_news_task'],
		)

	@task
	def website_scrape_task(self) -> Task:
		return Task(
			config=self.tasks_config['website_scrape_task'],
			#output_file='report.md'
		)
	
	@task
	def news_write_task(self) -> Task:
		return Task(
			config=self.tasks_config['news_write_task'],
			#output_file='report.md'
		)
	

	@task
	def file_write_task(self) -> Task:
		return Task(
			config=self.tasks_config['file_write_task'],
			#output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the AiSiteSummary crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
