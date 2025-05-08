using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace BackEnd.Models
{
	public class Question
	{
		public int Id { get; set; }
		public int TestId { get; set; }
		public string QuestionText { get; set; }
		public string QuestionType { get; set; }

		public Test Test { get; set; }
		public ICollection<AnswerOption> AnswerOptions { get; set; }
	}




}
