using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace BackEnd.Models
{
	public class AnswerOption
	{
		public int Id { get; set; }
		public int QuestionId { get; set; }
		public string AnswerText { get; set; }
		public bool IsCorrect { get; set; }

		public Question Question { get; set; }
	}


}
